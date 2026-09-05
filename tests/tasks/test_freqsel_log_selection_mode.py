"""The log has to name the selection the run actually uses.

``PoolScheduler`` has two node-selection modes and they are not
interchangeable. With ``random_batch`` set, ``__post_init__`` overwrites the
configured ``n_pools`` with ``ceil(n_cells / random_batch)`` and ``draw``
then ignores ``self.pools`` entirely, sampling a fresh random subset every
iteration. The stage header still printed "N pools" and the per-iteration line
still printed ``it % n_pools`` -- iteration parity dressed up as a selection.
A production cascade logged "2 pools" and "pool 0 / pool 1" for a whole run while
drawing a fresh random 470 of 500 cells every single iteration.
"""
from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pytest

from sweep_tasks.freqsel import PoolScheduler
from sweep_tasks.tasks import fwi_freqsel as FF


def _grid(n):
    """n distinct source cells, one per column."""
    return np.stack([np.zeros(n, int), np.arange(n), np.zeros(n, int)], 1)


def _sched(n_cells, n_pools=14, n_bins=470, random_batch=None):
    return PoolScheduler(node_grid=_grid(n_cells), n_pools=n_pools,
                         n_bins=n_bins, seed=0, random_batch=random_batch)


# --- rotation mode: the pool index is real, keep reporting it ---------------
def test_rotation_reports_pools():
    s = _sched(500, n_pools=14)
    assert s.selection == "14 pools"
    assert s.iteration_label(0) == "pool  0"
    assert s.iteration_label(15) == "pool  1"


def test_rotation_label_tracks_the_pool_that_fires():
    s = _sched(500, n_pools=14)
    for it in (0, 1, 13, 14, 27):
        pool, _ = s.draw(it)
        idx = int(re.search(r"pool\s+(\d+)", s.iteration_label(it)).group(1))
        assert np.array_equal(pool, s.pools[idx]), \
            "the printed pool index must be the pool that actually fired"


# --- random-batch mode: there is no pool index to report -------------------
def test_random_batch_reports_the_batch():
    s = _sched(500, n_pools=14, random_batch=470)
    assert s.n_cells == 500
    assert s.selection == "batch 470/500"
    assert "pool" not in s.selection
    assert "pool" not in s.iteration_label(0)
    assert "470" in s.iteration_label(0)


def test_random_batch_label_does_not_alternate():
    """The old field was `it % n_pools`, which alternated 0,1,0,1 forever."""
    s = _sched(500, n_pools=14, random_batch=470)
    labels = {s.iteration_label(it) for it in range(30)}
    assert len(labels) == 1, f"label must not vary with iteration: {labels}"


def test_random_batch_really_ignores_the_pools():
    """Guards the premise: if draw() ever starts using pools again, the label
    rule above needs revisiting rather than silently going stale."""
    s = _sched(500, n_pools=14, random_batch=470)
    pools = {tuple(p.tolist()) for p in s.pools}
    draws = {tuple(s.draw(i)[0].tolist()) for i in range(8)}
    assert not (draws & pools), "draw() returned a fixed pool in random mode"
    assert all(len(d) == 470 for d in draws)


def test_header_and_iteration_line_use_the_accessors():
    """Two call sites drifted apart once; keep them on one source of truth."""
    src = inspect.getsource(FF)
    tree = ast.parse(src)
    fstrings = [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    joined = "\n".join(fstrings)
    assert "sched.selection" in src
    assert "iteration_label" in src
    assert "n_pools" not in joined, \
        "no printed field may compute the selection from n_pools itself"
