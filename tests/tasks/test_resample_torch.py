"""GPU polyphase resample must equal scipy's resample_poly."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep_tasks.preproc.resample import (  # noqa: E402
    resample_time,
    resample_time_torch,
)

CASES = [
    (0.002, 0.0068),   # a 5/17 ratio
    (0.002, 0.004),    # plain 2x decimation
    (0.004, 0.002),    # upsample
    (0.001, 0.0017),   # awkward ratio
]


@pytest.mark.parametrize("dt_in,dt_out", CASES)
def test_matches_scipy(dt_in, dt_out):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 501)).astype(np.float64)
    ref = resample_time(x, dt_in, dt_out, axis=-1)
    got = resample_time_torch(torch.as_tensor(x), dt_in, dt_out, axis=-1).numpy()
    assert got.shape == ref.shape
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10


def test_matches_scipy_on_real_shaped_batch():
    """Shape of the production prefetch batch: (B, nrec, nt)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 40, 4001)).astype(np.float64)
    ref = resample_time(x, 0.002, 0.0068, axis=-1)
    got = resample_time_torch(torch.as_tensor(x), 0.002, 0.0068, axis=-1).numpy()
    assert got.shape == ref.shape == (3, 40, 1177)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10


def test_float32_input_stays_float32():
    x = torch.randn(4, 256)
    y = resample_time_torch(x, 0.002, 0.0068)
    assert y.dtype == torch.float32
    ref = resample_time(x.double().numpy(), 0.002, 0.0068, axis=-1)
    assert np.linalg.norm(y.numpy() - ref) / np.linalg.norm(ref) < 1e-6


def test_axis_argument():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 5))
    ref = resample_time(x, 0.002, 0.0068, axis=0)
    got = resample_time_torch(torch.as_tensor(x), 0.002, 0.0068, axis=0).numpy()
    assert got.shape == ref.shape
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10


def test_identity_ratio_is_a_passthrough():
    x = torch.randn(2, 64)
    assert resample_time_torch(x, 0.002, 0.002) is x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_scipy():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 4001)).astype(np.float32)
    ref = resample_time(x.astype(np.float64), 0.002, 0.0068, axis=-1)
    got = resample_time_torch(torch.as_tensor(x).cuda(), 0.002, 0.0068).cpu().numpy()
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6


def test_antialias_beats_linear_interpolation():
    """The reason polyphase is used at all: linear interp aliases."""
    dt_in, dt_out = 0.002, 0.0068
    n = 2001
    t = np.arange(n) * dt_in
    # a tone above the OUTPUT Nyquist (73.5 Hz) — must not survive decimation
    x = np.sin(2 * np.pi * 145.0 * t)[None, :]
    poly = resample_time_torch(torch.as_tensor(x), dt_in, dt_out).numpy()
    t_out = np.arange(poly.shape[-1]) * dt_out
    lin = np.interp(t_out, t, x[0])[None, :]
    # Away from the filter's start/end transients the tone is gone; linear
    # interpolation folds it straight down into the retained band.
    core = slice(60, -60)
    assert np.abs(poly[:, core]).max() < 0.02 * np.abs(x).max()
    assert np.abs(lin[:, core]).max() > 20 * np.abs(poly[:, core]).max()
