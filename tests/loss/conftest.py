"""Shared pytest fixtures for fwiloss tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def small_gather():
    """Return a synthetic gather (syn, obs) with shape (ns, nt, nrec, nchan).

    The traces are Ricker wavelets centred at slightly different lag times so
    every misfit has something non-trivial to chew on.
    """
    ns, nt, nrec, nchan = 2, 256, 8, 1
    dt = 1.0e-3
    t = torch.arange(nt) * dt
    fc = 15.0

    def ricker(t0):
        x = (np.pi * fc * (t - t0)) ** 2
        return (1 - 2 * x) * torch.exp(-x)

    obs = torch.zeros(ns, nt, nrec, nchan)
    syn = torch.zeros(ns, nt, nrec, nchan)
    for s in range(ns):
        for r in range(nrec):
            obs_lag = 0.05 + 0.001 * r + 0.01 * s
            syn_lag = obs_lag + 0.002 + 0.0005 * r
            obs[s, :, r, 0] = ricker(obs_lag)
            syn[s, :, r, 0] = ricker(syn_lag)
    return syn, obs, dt
