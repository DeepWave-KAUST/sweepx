"""The stage-entry obs bandpass is filtered a few shots at a time.

``_prepare_stage`` band-passes the whole obs on the GPU at every stage entry.
It used to filter 64 shots per call, which on the Marmousi examples (114 shots
of 10000 x 1361 samples) is 3.2 GiB of input and ~23 GiB of FFT transients: an
OOM on a 32 GB card before the first epoch. The chunk is now sized in bytes.
Filtering is per shot along the time axis, so the chunking must not change the
result.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweepx.tasks._helpers import stages
from sweepx.tasks._helpers.stages import (
    _OBS_BANDPASS_CHUNK_BYTES,
    _bandpass_obs_by_shot_chunks,
)
from sweepx.tasks.preproc.filter import bandpass_torch

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_KW = dict(lo=2.0, hi=12.0, dt=0.004, order=4)


def _obs(device, nshots=7, nt=500, nrec=33):
    g = torch.Generator().manual_seed(0)
    return torch.randn(nshots, nt, nrec, 1, generator=g).to(device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shots_per_chunk", [1, 3, 7, 50])
def test_chunked_bandpass_equals_one_call(device, shots_per_chunk):
    obs = _obs(device)
    want = bandpass_torch(obs.clone(), axis=1, **_KW)
    shot_bytes = obs[0].numel() * obs.element_size()
    got = _bandpass_obs_by_shot_chunks(
        obs.clone(), time_axis=1, chunk_bytes=shots_per_chunk * shot_bytes, **_KW)
    # same per-shot filter; only float32 round-off may differ between FFT batch sizes
    scale = float(want.abs().max())
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6 * scale)


def test_chunks_are_bounded_in_bytes_not_shots(monkeypatch):
    """At the examples' scale (114 shots x 10000 x 1361 fp32) a chunk is 9 shots,
    under 512 MiB, not 64 shots (3.2 GiB)."""
    seen = []

    def _record(x, **kw):
        seen.append(tuple(x.shape))
        return x

    monkeypatch.setattr(stages, "_bandpass_torch_fft", _record)
    obs = torch.empty(114, 10000, 1361, 1, device="meta")
    _bandpass_obs_by_shot_chunks(obs, time_axis=1, **_KW)
    shot_bytes = 10000 * 1361 * 4
    assert sum(s[0] for s in seen) == 114
    assert max(s[0] for s in seen) == _OBS_BANDPASS_CHUNK_BYTES // shot_bytes == 9
    assert all(s[0] * shot_bytes <= _OBS_BANDPASS_CHUNK_BYTES for s in seen)


def test_a_shot_larger_than_the_chunk_is_filtered_alone(monkeypatch):
    seen = []
    monkeypatch.setattr(stages, "_bandpass_torch_fft",
                        lambda x, **kw: seen.append(x.shape[0]) or x)
    obs = torch.empty(3, 100, 10, 1, device="meta")
    _bandpass_obs_by_shot_chunks(obs, time_axis=1, chunk_bytes=8, **_KW)
    assert seen == [1, 1, 1]


def test_time_on_the_first_axis_is_filtered_in_one_call():
    obs = _obs("cpu")[0, :, :, 0]                 # (nt, nrec): no shot axis to split
    want = bandpass_torch(obs.clone(), axis=0, **_KW)
    got = _bandpass_obs_by_shot_chunks(obs.clone(), time_axis=0, chunk_bytes=1, **_KW)
    np.testing.assert_array_equal(got.numpy(), want.numpy())
