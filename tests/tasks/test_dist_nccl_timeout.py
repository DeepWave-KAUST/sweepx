"""The process group must outlive a legitimate stage start.

NCCL's watchdog default is 10 minutes. Building the observed-coefficient table
streams hundreds of GB and the per-shard scale reduction sits inside that
window, so a band that takes ~16 minutes to stage is not hung -- but the
watchdog aborts every rank as if it were, and the run dies with
``WorkNCCL(...) ran for 600072 milliseconds before timing out`` after hours of
useful work.

``fwi_freqsel`` already read ``SWEEP_DD_NCCL_TIMEOUT_S`` for its own re-init;
the main entry point ignored it, so the knob only covered one of the two paths.
"""
from datetime import timedelta

import pytest

import sweep_tasks.runtime.distributed as D


@pytest.fixture
def spy(monkeypatch):
    """Pretend we are under torchrun with no group yet, and record the call."""
    seen = {}

    def fake_init(**kw):
        seen.update(kw)

    monkeypatch.setattr(D, "is_torchrun", lambda: True)
    monkeypatch.setattr(D.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(D.dist, "init_process_group", fake_init)
    return seen


def test_default_timeout_is_well_past_the_nccl_watchdog(spy, monkeypatch):
    monkeypatch.delenv("SWEEP_DD_NCCL_TIMEOUT_S", raising=False)
    D.init_process_group()
    assert spy["timeout"] == timedelta(seconds=1800)
    assert spy["timeout"] > timedelta(seconds=600), "must beat the NCCL default"


def test_env_overrides_it(spy, monkeypatch):
    monkeypatch.setenv("SWEEP_DD_NCCL_TIMEOUT_S", "5400")
    D.init_process_group()
    assert spy["timeout"] == timedelta(seconds=5400)


def test_backend_still_forwarded(spy, monkeypatch):
    monkeypatch.delenv("SWEEP_DD_NCCL_TIMEOUT_S", raising=False)
    D.init_process_group(backend="gloo")
    assert spy["backend"] == "gloo"


def test_no_op_when_not_under_torchrun(monkeypatch):
    calls = []
    monkeypatch.setattr(D, "is_torchrun", lambda: False)
    monkeypatch.setattr(D.dist, "init_process_group", lambda **kw: calls.append(kw))
    D.init_process_group()
    assert calls == []


def test_no_op_when_already_initialised(monkeypatch):
    calls = []
    monkeypatch.setattr(D, "is_torchrun", lambda: True)
    monkeypatch.setattr(D.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(D.dist, "init_process_group", lambda **kw: calls.append(kw))
    D.init_process_group()
    assert calls == []


# ---------------------------------------------------------------------------
# init_distributed_if_needed is the entry point every caller in the package
# actually reaches -- runner.py, fwi, rtm, lsrtm, fwi_plan_streaming all call
# it, and nothing outside this file calls init_process_group(). The first
# round of this fix patched only the latter, so production kept running on
# NCCL's 600 s default: a production run printed
# ``WorkNCCL(..., Timeout(ms)=600000)`` with SWEEP_DD_NCCL_TIMEOUT_S=5400 set
# and verified live in the process environment.
# ---------------------------------------------------------------------------
@pytest.fixture
def spy_if_needed(monkeypatch):
    """Pretend we are rank 0 of a 4-rank torchrun with no group yet."""
    seen = {}

    def fake_init(**kw):
        seen.update(kw)

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(D.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(D.dist, "init_process_group", fake_init)
    monkeypatch.setattr(D.torch.cuda, "is_available", lambda: False)
    return seen


def test_if_needed_default_timeout_beats_the_watchdog(spy_if_needed, monkeypatch):
    monkeypatch.delenv("SWEEP_DD_NCCL_TIMEOUT_S", raising=False)
    D.init_distributed_if_needed()
    assert spy_if_needed["timeout"] == timedelta(seconds=1800)
    assert spy_if_needed["timeout"] > timedelta(seconds=600)


def test_if_needed_env_overrides_it(spy_if_needed, monkeypatch):
    monkeypatch.setenv("SWEEP_DD_NCCL_TIMEOUT_S", "5400")
    D.init_distributed_if_needed()
    assert spy_if_needed["timeout"] == timedelta(seconds=5400)


def test_if_needed_backend_env_still_honoured(spy_if_needed, monkeypatch):
    monkeypatch.delenv("SWEEP_DD_NCCL_TIMEOUT_S", raising=False)
    monkeypatch.setenv("SWEEP_DIST_BACKEND", "gloo")
    D.init_distributed_if_needed(backend_env="SWEEP_DIST_BACKEND")
    assert spy_if_needed["backend"] == "gloo"


def test_if_needed_single_rank_creates_no_group(monkeypatch):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(D.dist, "init_process_group",
                        lambda **kw: calls.append(kw))
    info = D.init_distributed_if_needed()
    assert calls == [] and info.world_size == 1 and not info.initialised


def test_every_production_caller_uses_if_needed():
    """The knob is worthless if it sits on a function nobody calls."""
    import pathlib
    import re
    pkg = pathlib.Path(D.__file__).parent.parent
    # `<something>.init_process_group(` where <something> is not torch.dist
    call = re.compile(r"(?<!dist)\.init_process_group\(")
    hits = sorted(f.relative_to(pkg).as_posix() for f in pkg.rglob("*.py")
                  if f.name != "distributed.py" and call.search(f.read_text()))
    assert hits == [], f"these bypass init_distributed_if_needed: {hits}"
