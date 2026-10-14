"""A frequency ladder has to BE a ladder: each stage starts from the last."""
import numpy as np
import pytest

def test_grid_mode_stage_carries_the_previous_model():
    """A grid-parametrised freqsel ladder must start stage N from stage N-1.

    The stage setup rebuilt the Adam state on every entry and used to
    rebuild the MODEL from the pristine init alongside it, which silently made
    every rung but the last a no-op — the misfit fell, then the next stage threw
    the result away. Asserted against the real source: the si>0 grid branch must
    consult the carried model, not clone the base unconditionally.
    """
    import inspect
    import re

    from sweep_tasks.tasks.fwi_freqsel import FreqselRunnerMixin

    src = inspect.getsource(FreqselRunnerMixin._freqsel_run_stage)
    # the grid-vp branch of the si>0 (`else:`) arm
    tail = src[src.index("# grid-vp mode"):]
    tail = tail[:tail.index("# per-stage lr scale")]
    assert "_freqsel_last_vp" in tail, (
        "the si>0 grid branch ignores the previous stage's model:\n" + tail)
    # and it must still fall back to the base when there is nothing to carry
    assert re.search(r"base_t\.clone\(\)", tail), (
        "no fallback to the init model for the first stage:\n" + tail)


def test_freqsel_clears_carried_model_between_runs():
    """Two tasks through one runner instance must not share a starting model."""
    import inspect

    from sweep_tasks.tasks.fwi_freqsel import FreqselRunnerMixin

    src = inspect.getsource(FreqselRunnerMixin._run_fwi_freqsel)
    assert "self._freqsel_last_vp = None" in src, (
        "_run_fwi_freqsel must clear the carried model so a reused runner "
        "cannot leak one task's result into the next")


def test_carried_model_stays_a_leaf_across_a_grid_change():
    """Adam only steps a LEAF. Resampling then cloning must keep it one.

    Only a ladder that changes ``dh_m`` between stages walks this path, which
    is exactly why it is worth pinning: the shipped examples keep dh fixed.
    """
    import torch

    from sweep_tasks._helpers.stage_grid import _resample_to_grid, _stage_grid_shape

    prev = torch.full((8, 10), 1500.0, requires_grad=True)      # stage N-1 result

    # same grid: detach first, so the clone is a leaf
    a = prev.detach()
    vp_same = a.clone().requires_grad_(True)
    assert vp_same.is_leaf

    # grid change (20 m -> 10 m): the resample hands back a plain tensor, so
    # the clone is a leaf Adam can step
    shape = _stage_grid_shape((8, 10), 20.0, 10.0)
    b = _resample_to_grid(prev, 20.0, shape, 10.0)
    assert not b.requires_grad
    vp_resampled = b.to(prev.device).clone().requires_grad_(True)
    assert vp_resampled.is_leaf, "carried model is not a leaf; Adam cannot step it"
    assert vp_resampled.shape == (15, 19)

    # and the trap the old helper set: a requires_grad result cloned is NOT a leaf
    assert not prev.clone().is_leaf
