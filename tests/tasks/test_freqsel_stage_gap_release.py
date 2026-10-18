"""The card has to go back to the driver between stages.

Three runs of a multi-stage cascade died at the same place -- entering the
stage after the last one that keeps its boundary on the card, with
``ncclUnhandledCudaError: Call to CUDA function failed / Cuda failure 2 'out
of memory'``. The incoming stage needs about half of an 80 GB card; the stage
before it left most of the card sitting in PyTorch's caching allocator. NCCL's
communicator buffers for the new stage's process groups are allocated outside
that pool, so they got nothing.

The error names neither the allocation nor the stage, and a 600 s watchdog
timeout follows it, which is why this looked like a hang for two rounds.
"""
from __future__ import annotations

import ast
import inspect
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sweep_tasks.tasks import fwi_freqsel as F


class _Cuda:
    """Stand-in for torch.cuda: records calls, reports a full card."""

    def __init__(self, available=True, free=(1 << 28)):
        self._available = available
        self._free = free
        self.empty_cache_calls = 0

    def is_available(self):
        return self._available

    def empty_cache(self):
        self.empty_cache_calls += 1
        self._free = 79 << 30

    def mem_get_info(self):
        return self._free, 80 << 30


@pytest.fixture
def cuda(monkeypatch):
    import torch
    fake = _Cuda()
    monkeypatch.setattr(torch, "cuda", fake)
    monkeypatch.delenv("SWEEP_FREQSEL_STAGE_RELEASE", raising=False)
    return fake


def test_release_hands_the_card_back(cuda, capsys):
    F._release_card_between_stages(si=4, rank=0)
    assert cuda.empty_cache_calls == 1
    out = capsys.readouterr().out
    assert "stage 4 gap" in out and "release=on" in out
    # both numbers reported: the before is what says whether the cache is the
    # thing starving NCCL, and it was missing from the first investigation
    assert "0.2 -> 79.0 / 80.0 GiB" in out


def test_env_can_disable_it(cuda, capsys, monkeypatch):
    monkeypatch.setenv("SWEEP_FREQSEL_STAGE_RELEASE", "0")
    F._release_card_between_stages(si=4, rank=0)
    assert cuda.empty_cache_calls == 0
    assert "release=OFF" in capsys.readouterr().out


def test_only_rank_zero_prints(cuda, capsys):
    F._release_card_between_stages(si=4, rank=1)
    assert cuda.empty_cache_calls == 1, "every rank releases"
    assert capsys.readouterr().out == "", "only rank 0 reports"


def test_no_cuda_is_a_no_op(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch, "cuda", _Cuda(available=False))
    F._release_card_between_stages(si=0, rank=0)
    assert capsys.readouterr().out == ""


def test_the_stage_loop_actually_calls_it():
    """A helper nobody calls fixes nothing -- see the NCCL-timeout round."""
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(F.FreqselRunnerMixin._run_fwi_freqsel)))
    loops = [n for n in ast.walk(tree)
             if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and [getattr(e, "id", None) for e in n.target.elts] == ["si", "stage"]]
    assert len(loops) == 1, "expected exactly one per-stage loop"
    called = {n.func.id for n in ast.walk(loops[0])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_release_card_between_stages" in called


# ---------------------------------------------------------------------------
# The reproduction. Needs two GPUs, so it skips nearly everywhere; when it
# does run it is the whole argument: fill the card, drop the references so
# the blocks land in the cache rather than with the driver, then do what a
# new stage does -- open a process group and use it.
# ---------------------------------------------------------------------------
_REPRO = r"""
import os, torch, torch.distributed as dist
GiB = 1024 ** 3
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
dist.all_reduce(torch.ones(1, device="cuda"))
blocks = []
try:
    while True:
        blocks.append(torch.empty(GiB, dtype=torch.uint8, device="cuda"))
except Exception:
    pass
del blocks
if os.environ["RELEASE"] != "0":
    torch.cuda.empty_cache()
try:
    pg = dist.new_group(ranks=list(range(dist.get_world_size())))
    dist.all_reduce(torch.ones(1, device="cuda"), group=pg)
    torch.cuda.synchronize()
    ok = True
except Exception:
    ok = False
if int(os.environ["RANK"]) == 0:
    print(f"NEWGROUP_OK={ok}")
"""


def _run_repro(tmp_path: Path, release: str) -> str:
    script = tmp_path / "repro.py"
    script.write_text(_REPRO)
    env = {**os.environ, "RELEASE": release,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    p = subprocess.run(
        [shutil.which("torchrun"), "--standalone", "--nproc_per_node=2",
         str(script)],
        env=env, capture_output=True, text=True, timeout=600)
    return p.stdout


@pytest.mark.skipif(shutil.which("torchrun") is None, reason="no torchrun")
def test_a_full_cache_is_what_breaks_nccl(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    assert "NEWGROUP_OK=False" in _run_repro(tmp_path, "0"), \
        "expected NCCL to fail while the allocator still owns the card"
    assert "NEWGROUP_OK=True" in _run_repro(tmp_path, "1"), \
        "expected the release to clear it"
