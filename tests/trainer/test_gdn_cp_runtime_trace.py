import hashlib
import json
import stat
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from veomni.distributed.context_parallel.gdn_runtime import (
    GdnCpOperation,
    GdnCpPhase,
    GdnCpRuntimeIdentity,
    GdnCpRuntimeObserver,
)
from veomni.trainer.callbacks import trace_callback


class _ObservedLayer(torch.nn.Module):
    def __init__(self, observer: GdnCpRuntimeObserver) -> None:
        super().__init__()
        self.gdn_cp_runtime_evidence = observer


def _state_observer(*, cp_size: int = 4, cp_rank: int = 1, include_backward_recv: bool = True) -> GdnCpRuntimeObserver:
    observer = GdnCpRuntimeObserver(
        GdnCpRuntimeIdentity(
            implementation="state_passing_lossless",
            ownership_plan_hash="plan-123",
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
    )
    for phase in (GdnCpPhase.FORWARD, GdnCpPhase.BACKWARD):
        operations = [GdnCpOperation.OWNERSHIP_A2A]
        if cp_rank > 0:
            if phase is not GdnCpPhase.BACKWARD or include_backward_recv:
                operations.extend((GdnCpOperation.STATE_P2P_RECV, GdnCpOperation.HALO_P2P_RECV))
        if cp_rank + 1 < cp_size:
            operations.extend((GdnCpOperation.STATE_P2P_SEND, GdnCpOperation.HALO_P2P_SEND))
        for operation in operations:
            observer.enter(operation, phase)
            observer.exit(operation, phase)
    observer.observe_cp_ranks(range(cp_size))
    return observer


def _kcp_observer(*, cp_size: int = 4, cp_rank: int = 1) -> GdnCpRuntimeObserver:
    observer = GdnCpRuntimeObserver(
        GdnCpRuntimeIdentity(
            implementation="kcp",
            ownership_plan_hash="plan-123",
            cp_size=cp_size,
            cp_rank=cp_rank,
            affine_backend="ttx_bc8_m1",
        )
    )
    observer.enter(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
    observer.exit(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
    for phase in (GdnCpPhase.FORWARD, GdnCpPhase.BACKWARD):
        operations = [GdnCpOperation.OWNERSHIP_A2A, GdnCpOperation.KCP_AFFINE_AG]
        if cp_rank > 0:
            operations.append(GdnCpOperation.HALO_P2P_RECV)
        if cp_rank + 1 < cp_size:
            operations.append(GdnCpOperation.HALO_P2P_SEND)
        for operation in operations:
            observer.enter(operation, phase)
            observer.exit(operation, phase)
    observer.observe_cp_ranks(range(cp_size))
    return observer


@pytest.mark.parametrize("value", ["", "-1", "1.0", "true"])
def test_gdn_cp_runtime_trace_rejects_invalid_step_limits(monkeypatch, value):
    monkeypatch.setenv("VEOMNI_GDN_CP_RUNTIME_TRACE_STEPS", value)
    with pytest.raises(ValueError, match="non-negative base-10 integer"):
        trace_callback._gdn_cp_runtime_trace_steps()


def test_gdn_cp_runtime_trace_proves_state_lossless_collectives():
    model = torch.nn.Sequential(_ObservedLayer(_state_observer()), _ObservedLayer(_state_observer()))

    trace = trace_callback._collect_gdn_cp_runtime_trace(model)

    assert trace["identity"] == {
        "implementation": "state_passing_lossless",
        "ownership_plan_hash": "plan-123",
        "cp_size": 4,
        "cp_rank": 1,
        "layout": "lossless_sparse_packed",
        "affine_backend": None,
    }
    assert trace["observer_count"] == 2
    assert trace["observed_cp_ranks"] == [0, 1, 2, 3]
    assert trace["balanced"] is True
    assert {
        (event["operation"], event["phase"], event["enter"], event["exit"], event["error"])
        for event in trace["operations"]
    } >= {
        ("ownership_a2a", "forward", 2, 2, 0),
        ("ownership_a2a", "backward", 2, 2, 0),
        ("state_p2p_recv", "forward", 2, 2, 0),
        ("state_p2p_send", "backward", 2, 2, 0),
        ("halo_p2p_recv", "backward", 2, 2, 0),
        ("halo_p2p_send", "forward", 2, 2, 0),
    }


def test_gdn_cp_runtime_trace_matches_live_cp_and_ulysses_groups(monkeypatch):
    cp_group = object()
    ulysses_group = object()
    sp_group = object()
    group_sizes = {cp_group: 4, ulysses_group: 2, sp_group: 8}
    group_ranks = {cp_group: 1, ulysses_group: 0, sp_group: 2}
    monkeypatch.setattr(dist, "get_world_size", lambda group: group_sizes[group])
    monkeypatch.setattr(dist, "get_rank", lambda group: group_ranks[group])
    parallel_state = SimpleNamespace(
        cp_enabled=True,
        cp_group=cp_group,
        cp_size=4,
        cp_rank=1,
        ulysses_enabled=True,
        ulysses_group=ulysses_group,
        ulysses_size=2,
        ulysses_rank=0,
        sp_group=sp_group,
        sp_size=8,
        sp_rank=2,
    )
    trace = trace_callback._collect_gdn_cp_runtime_trace(_ObservedLayer(_state_observer()))

    assert trace_callback._validate_gdn_cp_runtime_topology(trace, parallel_state) == {
        "cp_size": 4,
        "cp_rank": 1,
        "ulysses_size": 2,
        "ulysses_rank": 0,
        "sp_size": 8,
        "sp_rank": 2,
        "ulysses_group_size": 2,
        "sp_group_size": 8,
    }


def test_gdn_cp_runtime_trace_rejects_observer_topology_drift(monkeypatch):
    group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda unused_group: 4)
    monkeypatch.setattr(dist, "get_rank", lambda unused_group: 2)
    parallel_state = SimpleNamespace(
        cp_enabled=True,
        cp_group=group,
        cp_size=4,
        cp_rank=2,
        ulysses_enabled=False,
        ulysses_group=None,
        ulysses_size=1,
        ulysses_rank=-1,
        sp_group=group,
        sp_size=4,
        sp_rank=2,
    )
    trace = trace_callback._collect_gdn_cp_runtime_trace(_ObservedLayer(_state_observer(cp_rank=1)))

    with pytest.raises(RuntimeError, match="observer identity differs"):
        trace_callback._validate_gdn_cp_runtime_topology(trace, parallel_state)


def test_gdn_cp_runtime_trace_rejects_missing_backward_collective():
    model = _ObservedLayer(_state_observer(include_backward_recv=False))

    with pytest.raises(RuntimeError, match="missing a balanced collective"):
        trace_callback._collect_gdn_cp_runtime_trace(model)


def test_gdn_cp_runtime_trace_rejects_inactive_observer_masked_by_active_layer():
    inactive = GdnCpRuntimeObserver(_state_observer().identity)
    model = torch.nn.Sequential(_ObservedLayer(_state_observer()), _ObservedLayer(inactive))

    with pytest.raises(RuntimeError, match="per-observer observer"):
        trace_callback._collect_gdn_cp_runtime_trace(model)


def test_gdn_cp_runtime_trace_uses_per_optimizer_step_deltas():
    observer = _state_observer()
    model = _ObservedLayer(observer)
    previous = trace_callback._snapshot_gdn_cp_runtime_observers(model)

    with pytest.raises(RuntimeError, match="per-observer observer"):
        trace_callback._collect_gdn_cp_runtime_trace(model, previous_snapshots=previous)

    for phase in (GdnCpPhase.FORWARD, GdnCpPhase.BACKWARD):
        operations = [GdnCpOperation.OWNERSHIP_A2A]
        operations.extend((GdnCpOperation.STATE_P2P_RECV, GdnCpOperation.HALO_P2P_RECV))
        operations.extend((GdnCpOperation.STATE_P2P_SEND, GdnCpOperation.HALO_P2P_SEND))
        for operation in operations:
            observer.enter(operation, phase)
            observer.exit(operation, phase)

    trace = trace_callback._collect_gdn_cp_runtime_trace(model, previous_snapshots=previous)
    assert all(event["enter"] == 1 for event in trace["operations"])


def test_gdn_cp_runtime_trace_proves_kcp_operation_matrix():
    trace = trace_callback._collect_gdn_cp_runtime_trace(_ObservedLayer(_kcp_observer()))
    assert trace["identity"]["implementation"] == "kcp"


def test_gdn_cp_runtime_trace_rejects_kcp_without_halo_or_with_state_p2p():
    missing_halo = _kcp_observer()
    model = _ObservedLayer(missing_halo)
    previous = trace_callback._snapshot_gdn_cp_runtime_observers(model)
    for phase in (GdnCpPhase.FORWARD, GdnCpPhase.BACKWARD):
        for operation in (GdnCpOperation.OWNERSHIP_A2A, GdnCpOperation.KCP_AFFINE_AG):
            missing_halo.enter(operation, phase)
            missing_halo.exit(operation, phase)
    with pytest.raises(RuntimeError, match="halo_p2p_recv"):
        trace_callback._collect_gdn_cp_runtime_trace(model, previous_snapshots=previous)

    mixed = _kcp_observer()
    mixed.enter(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.FORWARD)
    mixed.exit(GdnCpOperation.STATE_P2P_RECV, GdnCpPhase.FORWARD)
    with pytest.raises(RuntimeError, match="kcp unexpectedly executed state_p2p_recv"):
        trace_callback._collect_gdn_cp_runtime_trace(_ObservedLayer(mixed))


def test_gdn_cp_runtime_trace_rejects_cp1_and_missing_observers():
    with pytest.raises(RuntimeError, match="no live context-parallel observers"):
        trace_callback._collect_gdn_cp_runtime_trace(torch.nn.Linear(2, 2))

    model = _ObservedLayer(_state_observer(cp_size=1, cp_rank=0))
    with pytest.raises(RuntimeError, match="requires a real CP topology"):
        trace_callback._collect_gdn_cp_runtime_trace(model)


def test_gdn_cp_runtime_trace_persists_one_private_atomic_artifact(tmp_path):
    trace = {
        "global_rank": 6,
        "global_step": 1,
        "identity": {"implementation": "state_passing_lossless"},
        "observers": [{"module": "layers.0"}],
    }

    reference = trace_callback._persist_gdn_cp_runtime_trace(trace, tmp_path)

    artifact = tmp_path / "step_00000001_rank_00006.json"
    assert reference == {
        "bytes": artifact.stat().st_size,
        "global_rank": 6,
        "global_step": 1,
        "sha256": reference["sha256"],
    }
    assert len(reference["sha256"]) == 64
    assert reference["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert json.loads(artifact.read_text(encoding="utf-8")) == trace
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("global_rank", True), ("global_rank", -1), ("global_step", True), ("global_step", 0)],
)
def test_gdn_cp_runtime_trace_artifact_rejects_invalid_identity(tmp_path, field, value):
    trace = {"global_rank": 0, "global_step": 1}
    trace[field] = value

    with pytest.raises(ValueError, match=field):
        trace_callback._persist_gdn_cp_runtime_trace(trace, tmp_path)


def test_gdn_cp_runtime_trace_artifact_refuses_to_overwrite_prior_attempt(tmp_path):
    trace = {"global_rank": 0, "global_step": 1}
    trace_callback._persist_gdn_cp_runtime_trace(trace, tmp_path)

    with pytest.raises(RuntimeError, match="already exists"):
        trace_callback._persist_gdn_cp_runtime_trace(trace, tmp_path)


def test_gdn_cp_runtime_trace_emits_short_reference_when_artifact_dir_is_set(monkeypatch, tmp_path, capfd):
    trace = {"global_rank": 2, "global_step": 1, "observers": [{"module": "layers.0"}]}
    monkeypatch.setenv("VEOMNI_GDN_CP_RUNTIME_TRACE_DIR", str(tmp_path))

    trace_callback._emit_gdn_cp_runtime_trace(trace)

    output = capfd.readouterr().out
    prefix = "VEOMNI_GDN_CP_RUNTIME_TRACE_REF "
    assert output.startswith(prefix)
    assert len(output.encode("utf-8")) < 4096
    reference = json.loads(output.removeprefix(prefix))
    assert reference["global_rank"] == 2
    assert reference["global_step"] == 1
    assert (tmp_path / "step_00000001_rank_00002.json").is_file()


def test_gdn_cp_runtime_trace_keeps_legacy_stdout_without_artifact_dir(monkeypatch, capsys):
    trace = {"global_rank": 2, "global_step": 1}
    monkeypatch.delenv("VEOMNI_GDN_CP_RUNTIME_TRACE_DIR", raising=False)

    trace_callback._emit_gdn_cp_runtime_trace(trace)

    output = capsys.readouterr().out
    assert output.startswith("VEOMNI_GDN_CP_RUNTIME_TRACE ")
    assert json.loads(output.split(" ", 1)[1]) == trace
