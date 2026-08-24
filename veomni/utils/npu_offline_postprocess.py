# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline Ascend profile post-processing (copy / analyse / upload).

Designed to run outside the training critical path. Training can spawn this as a
fire-and-forget sidecar after raw `*_ascend_pt` finalization so peer ranks are
not blocked on Chrome/DB export or Merlin upload.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional


logger = logging.getLogger("veomni.npu_offline_postprocess")

SIDECAR_LOG_BASENAME = "veomni_npu_offline_postprocess.log"
RAW_COMPLETION_MARKER_SUFFIX = "host/end_info.done"
DEFAULT_RAW_READY_TIMEOUT_SECONDS = 300.0
DEFAULT_RAW_STABLE_SECONDS = 5.0
DEFAULT_RAW_POLL_INTERVAL_SECONDS = 1.0

RawDirectorySnapshot = tuple[tuple[str, str, int, int, int], ...]


def _is_ascend_pt_dir(path: Path) -> bool:
    return path.is_dir() and path.name.endswith("_ascend_pt")


def resolve_raw_dir(raw_dir: str | Path) -> Path:
    """Return the concrete `*_ascend_pt` directory."""
    path = Path(raw_dir).expanduser().resolve()
    if _is_ascend_pt_dir(path):
        return path
    if path.is_dir():
        matches = sorted(p for p in path.iterdir() if _is_ascend_pt_dir(p))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple *_ascend_pt directories under {path}; pass one explicitly: {[p.name for p in matches]}"
            )
    raise FileNotFoundError(f"No *_ascend_pt directory found at {raw_dir}")


def resolve_analyse_path(raw_dir: Path) -> str:
    """Target exactly one finalized ``*_ascend_pt`` capture."""
    return str(raw_dir)


def snapshot_raw_dir(raw_dir: Path) -> RawDirectorySnapshot:
    """Return a recursive metadata snapshot of one raw Ascend capture.

    New sidecars write their log next to this tree. A legacy log found inside
    the raw directory remains part of the snapshot, so an older sidecar that is
    still appending to it cannot make a recursive copy look quiescent.
    """
    root = raw_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw Ascend profile directory does not exist: {root}")

    entries: list[tuple[str, str, int, int, int]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(f"Raw Ascend profile changed while scanning {directory}: {exc}") from exc
        for child in children:
            relative = child.path.removeprefix(f"{root}{os.sep}")
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"Raw Ascend profile changed while scanning {child.path}: {exc}") from exc

            if stat.S_ISDIR(info.st_mode):
                entry_type = "dir"
                pending.append(Path(child.path))
            elif stat.S_ISREG(info.st_mode):
                entry_type = "file"
            elif stat.S_ISLNK(info.st_mode):
                entry_type = "symlink"
            else:
                entry_type = "other"
            entries.append((relative, entry_type, info.st_size, info.st_mtime_ns, info.st_ctime_ns))

    # Each directory is scanned in a deterministic order, so a second global
    # O(N log N) sort is unnecessary for large captures.
    return tuple(entries)


def _completion_markers(snapshot: RawDirectorySnapshot) -> tuple[str, ...]:
    """Return the completion markers required by this CANN raw-tree layout.

    Current CANN writes one marker below every top-level ``PROF_*`` directory.
    The root-level form is retained for older/synthetic captures.
    """
    profile_roots = sorted(
        entry[0] for entry in snapshot if entry[1] == "dir" and os.sep not in entry[0] and entry[0].startswith("PROF_")
    )
    if profile_roots:
        return tuple(f"{profile_root}/{RAW_COMPLETION_MARKER_SUFFIX}" for profile_root in profile_roots)
    return (RAW_COMPLETION_MARKER_SUFFIX,)


def _validate_wait_seconds(name: str, value: float, *, allow_zero: bool) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    lower_bound_ok = normalized >= 0.0 if allow_zero else normalized > 0.0
    if not math.isfinite(normalized) or not lower_bound_ok:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {relation}, got {value!r}")
    return normalized


def wait_for_raw_dir_stable(
    raw_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_RAW_READY_TIMEOUT_SECONDS,
    stable_seconds: float = DEFAULT_RAW_STABLE_SECONDS,
    poll_interval_seconds: float = DEFAULT_RAW_POLL_INTERVAL_SECONDS,
) -> RawDirectorySnapshot:
    """Wait for the Ascend completion marker and a bounded quiet window.

    ``torch_npu`` can return from profiler finalization before late CANN files
    (notably ``host/end_info.done`` and hash dictionaries) are visible. A
    recursive copy or analysis started in that interval observes a torn tree.
    This barrier is sidecar-local: it never blocks distributed training, but it
    fails closed after ``timeout_seconds`` instead of copying partial data.
    """
    timeout = _validate_wait_seconds("timeout_seconds", timeout_seconds, allow_zero=False)
    stable_for = _validate_wait_seconds("stable_seconds", stable_seconds, allow_zero=True)
    poll_interval = _validate_wait_seconds("poll_interval_seconds", poll_interval_seconds, allow_zero=False)

    resolved = raw_dir.resolve()
    deadline = time.monotonic() + timeout
    stable_since: Optional[float] = None
    last_snapshot: Optional[RawDirectorySnapshot] = None
    last_reason = f"missing completion marker */{RAW_COMPLETION_MARKER_SUFFIX}"

    while True:
        now = time.monotonic()
        try:
            snapshot = snapshot_raw_dir(resolved)
        except (OSError, RuntimeError) as exc:
            snapshot = None
            last_reason = str(exc)

        if snapshot is not None:
            entry_by_path = {entry[0]: entry for entry in snapshot}
            required_markers = _completion_markers(snapshot)
            missing_markers = [
                marker_path
                for marker_path in required_markers
                if (marker := entry_by_path.get(marker_path)) is None or marker[1] != "file"
            ]
            payload_files = [
                entry
                for entry in snapshot
                if entry[1] == "file" and entry[0] not in {*required_markers, SIDECAR_LOG_BASENAME}
            ]
            if missing_markers:
                last_reason = f"missing completion marker(s): {', '.join(missing_markers)}"
                last_snapshot = None
                stable_since = None
            elif not payload_files:
                last_reason = "completion marker exists but the raw capture has no payload files"
                last_snapshot = None
                stable_since = None
            elif snapshot != last_snapshot:
                last_snapshot = snapshot
                stable_since = now
                last_reason = "raw capture is still changing"
            elif stable_since is not None and now - stable_since >= stable_for:
                file_count = sum(entry[1] == "file" for entry in snapshot)
                total_bytes = sum(entry[2] for entry in snapshot if entry[1] == "file")
                logger.info(
                    "NPU_PROFILE_RAW_READY raw_dir=%s stable_seconds=%.3f files=%d bytes=%d",
                    resolved,
                    now - stable_since,
                    file_count,
                    total_bytes,
                )
                return snapshot

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"Raw Ascend profile did not become ready at {resolved} within {timeout:.1f}s: {last_reason}. "
                "No copy or analysis was attempted."
            )
        time.sleep(min(poll_interval, remaining))


def _assert_raw_snapshot(raw_dir: Path, expected_snapshot: RawDirectorySnapshot, phase: str) -> None:
    observed = snapshot_raw_dir(raw_dir)
    if observed != expected_snapshot:
        raise RuntimeError(f"Raw Ascend profile changed {phase}: {raw_dir}. No further postprocess action is safe.")


def _publish_hdfs_staging(staging: str, target: str) -> None:
    hdfs_binary = shutil.which("hdfs")
    if hdfs_binary is None:
        raise RuntimeError(f"Cannot publish staged HDFS profile: hdfs executable is unavailable (staging={staging})")
    subprocess.run([hdfs_binary, "dfs", "-mv", staging, target], check=True)


def find_trace_view(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json",
        raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(raw_dir.rglob("trace_view.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"trace_view.json not found under {raw_dir} after analyse")


def gzip_trace(trace_path: Path) -> Path:
    if trace_path.suffix == ".gz":
        return trace_path
    out = Path(f"{trace_path}.gz")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(trace_path, "rb") as src, gzip.open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def copy_raw_dir(
    raw_dir: Path,
    copy_to: str,
    *,
    expected_snapshot: Optional[RawDirectorySnapshot] = None,
) -> None:
    if expected_snapshot is not None:
        _assert_raw_snapshot(raw_dir, expected_snapshot, "after the readiness barrier")

    if copy_to.startswith("hdfs://"):
        try:
            from veomni.utils.hdfs_io import copy, exists, makedirs
        except Exception as exc:  # pragma: no cover - import path depends on install
            raise RuntimeError(f"HDFS copy requires veomni.utils.hdfs_io: {exc}") from exc
        destination = copy_to.rstrip("/")
        target = f"{destination}/{raw_dir.name}"
        if exists(target):
            raise FileExistsError(f"Refusing to overwrite existing HDFS profile target: {target}")
        makedirs(destination, exist_ok=True)
        staging = f"{destination}/.{raw_dir.name}.incomplete-{uuid.uuid4().hex}"
        if not copy(str(raw_dir), staging):
            raise RuntimeError(f"Failed to stage {raw_dir} at {staging}; the final target {target} was not published")
        if expected_snapshot is not None:
            _assert_raw_snapshot(raw_dir, expected_snapshot, "while copying to HDFS")
        if exists(target):
            raise FileExistsError(
                f"Refusing to publish staged HDFS profile because the final target appeared concurrently: {target}. "
                f"Incomplete staging remains at {staging}."
            )
        _publish_hdfs_staging(staging, target)
        logger.info("Copied raw profile to %s", target)
        return

    source = raw_dir.resolve()
    dest = Path(copy_to).expanduser().resolve()
    target = dest / raw_dir.name
    if target == source:
        raise ValueError(f"copy_to cannot be the raw profile source parent: {dest}")
    try:
        dest.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(f"copy_to cannot be inside the raw profile directory: {dest}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing profile target: {target} already exists")
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / f".{raw_dir.name}.incomplete-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        if expected_snapshot is not None:
            _assert_raw_snapshot(raw_dir, expected_snapshot, "while copying locally")
        if target.exists():
            raise FileExistsError(
                f"Refusing to publish staged profile because the final target appeared concurrently: {target}"
            )
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("Copied raw profile to %s", target)


def run_analyse(raw_dir: Path, max_process_number: Optional[int] = None) -> None:
    try:
        from torch_npu.profiler.profiler import analyse as npu_analyse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch_npu is required for --analyse") from exc

    profiler_path = resolve_analyse_path(raw_dir)
    trace_path = raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if trace_path.is_file():
        existing_stat = trace_path.stat()
        existing_signature = (
            existing_stat.st_ino,
            existing_stat.st_size,
            existing_stat.st_mtime_ns,
            existing_stat.st_ctime_ns,
        )
    else:
        existing_signature = None
    kwargs = {}
    if max_process_number is not None:
        kwargs["max_process_number"] = max_process_number
    logger.info("Running torch_npu.profiler.profiler.analyse(profiler_path=%s)", profiler_path)
    started = time.perf_counter()
    npu_analyse(profiler_path=profiler_path, **kwargs)
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace_view.json not found under {raw_dir} after analyse")
    trace_stat = trace_path.stat()
    if trace_stat.st_size == 0:
        raise RuntimeError(f"torch_npu analyse produced an empty trace: {trace_path}")
    trace_signature = (
        trace_stat.st_ino,
        trace_stat.st_size,
        trace_stat.st_mtime_ns,
        trace_stat.st_ctime_ns,
    )
    if trace_signature == existing_signature:
        raise RuntimeError(f"torch_npu analyse did not create or update the existing trace: {trace_path}")
    with open(trace_path, "rb") as trace_file:
        head = trace_file.read(4096).lstrip()
        trace_file.seek(max(0, trace_stat.st_size - 4096))
        tail = trace_file.read().rstrip()
    if not head.startswith(b"[") or not tail.endswith(b"]"):
        raise RuntimeError(f"torch_npu analyse produced an incomplete trace: {trace_path}")
    logger.info(
        "NPU_PROFILE_ANALYSE mode=offline duration_seconds=%.6f raw_dir=%s trace=%s",
        time.perf_counter() - started,
        raw_dir,
        trace_path,
    )


def run_upload_cmd(upload_cmd: str | list[str], trace_file: Path) -> None:
    if isinstance(upload_cmd, list):
        logger.info("Uploading with argv: %s", shlex.join(upload_cmd))
        subprocess.run(upload_cmd, check=True)
        return

    quoted_trace = shlex.quote(str(trace_file))
    if "{trace}" in upload_cmd:
        command = upload_cmd.replace("{trace}", quoted_trace)
    else:
        command = f"{upload_cmd} {quoted_trace}"
    logger.info("Uploading with command: %s", command)
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def build_merlin_upload_cmd(trace_file: Path, name: Optional[str] = None) -> list[str]:
    payload = {
        "file_path": str(trace_file),
        "asset_type": "perfetto",
        "compress": False if str(trace_file).endswith(".gz") else True,
    }
    if name:
        payload["name"] = name
    # The JobRun Profiling tab lists assets by its selected Arnold trial.
    job_id = os.getenv("RH2_JOB_RUN_ID") or os.getenv("MERLIN_JOB_ID")
    trial_id = os.getenv("ARNOLD_TRIAL_ID")
    if trial_id:
        payload["trial_id"] = trial_id
    elif job_id:
        payload["job_id"] = job_id
    return ["merlin-cli", "profiling", "upload", "--json", json.dumps(payload, ensure_ascii=False)]


def upload_merlin_profile(trace_file: Path, name: Optional[str] = None) -> None:
    """Upload through Merlin's file-based CLI and fail safely when unavailable."""
    if shutil.which("merlin-cli"):
        run_upload_cmd(build_merlin_upload_cmd(trace_file, name=name), trace_file)
        return

    raise RuntimeError(
        "merlin-cli is unavailable. The bytedmerlin ProfilingAsset SDK fallback is intentionally unsupported "
        "because its JSON/base64 upload is unsafe for large traces. No upload was attempted; the raw profile "
        f"and packed trace remain on disk (trace={trace_file})."
    )


def postprocess(
    raw_dir: str | Path,
    *,
    copy_to: Optional[str] = None,
    analyse: bool = False,
    upload_cmd: Optional[str] = None,
    merlin_upload: bool = False,
    upload_name: Optional[str] = None,
    max_process_number: Optional[int] = None,
    raw_ready_timeout_seconds: float = DEFAULT_RAW_READY_TIMEOUT_SECONDS,
    raw_stable_seconds: float = DEFAULT_RAW_STABLE_SECONDS,
    raw_poll_interval_seconds: float = DEFAULT_RAW_POLL_INTERVAL_SECONDS,
) -> Optional[Path]:
    resolved = resolve_raw_dir(raw_dir)
    logger.info("Post-processing Ascend raw profile at %s", resolved)

    stable_snapshot = None
    if copy_to or analyse:
        stable_snapshot = wait_for_raw_dir_stable(
            resolved,
            timeout_seconds=raw_ready_timeout_seconds,
            stable_seconds=raw_stable_seconds,
            poll_interval_seconds=raw_poll_interval_seconds,
        )

    if copy_to:
        copy_raw_dir(resolved, copy_to, expected_snapshot=stable_snapshot)

    if analyse:
        assert stable_snapshot is not None
        _assert_raw_snapshot(resolved, stable_snapshot, "before analysis")
        run_analyse(resolved, max_process_number=max_process_number)

    if not (upload_cmd or merlin_upload):
        return None

    if not analyse:
        # Allow upload of an already-parsed capture.
        logger.info("Upload requested without --analyse; expecting an existing trace_view.json")

    trace_path = find_trace_view(resolved)
    packed = gzip_trace(trace_path)
    if merlin_upload:
        upload_merlin_profile(packed, name=upload_name)
    else:
        assert upload_cmd is not None
        run_upload_cmd(upload_cmd, packed)
    return packed


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy / analyse / upload an Ascend offline raw profile outside training."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to *_ascend_pt or its parent directory")
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Durable destination for the raw directory (local path or hdfs://...)",
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="Run torch_npu.profiler.profiler.analyse to produce Chrome/DB outputs",
    )
    parser.add_argument(
        "--upload-cmd",
        default=None,
        help="Shell command to upload the parsed trace. Use {trace} or append the path.",
    )
    parser.add_argument(
        "--merlin-upload",
        action="store_true",
        help="Upload the parsed Chrome trace with the file-based Merlin CLI.",
    )
    parser.add_argument("--upload-name", default=None, help="Optional Merlin asset display name")
    parser.add_argument("--max-process-number", type=int, default=None, help="torch_npu analyse parallelism")
    parser.add_argument(
        "--raw-ready-timeout-seconds",
        type=float,
        default=DEFAULT_RAW_READY_TIMEOUT_SECONDS,
        help="Maximum time to wait for a complete, stable raw capture before copy/analyse",
    )
    parser.add_argument(
        "--raw-stable-seconds",
        type=float,
        default=DEFAULT_RAW_STABLE_SECONDS,
        help="Required unchanged-tree window after host/end_info.done appears",
    )
    parser.add_argument(
        "--raw-poll-interval-seconds",
        type=float,
        default=DEFAULT_RAW_POLL_INTERVAL_SECONDS,
        help="Polling interval while waiting for raw capture readiness",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)
    if not (args.copy_to or args.analyse or args.upload_cmd or args.merlin_upload):
        logger.error("Nothing to do: pass --copy-to and/or --analyse and/or an upload option")
        return 2
    postprocess(
        args.raw_dir,
        copy_to=args.copy_to,
        analyse=args.analyse,
        upload_cmd=args.upload_cmd,
        merlin_upload=args.merlin_upload,
        upload_name=args.upload_name,
        max_process_number=args.max_process_number,
        raw_ready_timeout_seconds=args.raw_ready_timeout_seconds,
        raw_stable_seconds=args.raw_stable_seconds,
        raw_poll_interval_seconds=args.raw_poll_interval_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
