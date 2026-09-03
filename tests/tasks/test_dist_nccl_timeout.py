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
