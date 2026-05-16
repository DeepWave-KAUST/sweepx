"""Tests for sweep_io.cuda_prefetch — skipped without torch / cuda."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable; skipping cuda_prefetch tests", allow_module_level=True)

from sweep_io.cuda_prefetch import CUDAPrefetcher  # noqa: E402


def test_cuda_prefetch_yields_device_tensors():
    def load(i: int) -> np.ndarray:
        return np.full((4, 8), float(i), dtype="float32")

    with CUDAPrefetcher((load(i) for i in range(5)), device="cuda:0") as pf:
        out = list(pf)
    assert len(out) == 5
    for i, t in enumerate(out):
        assert isinstance(t, torch.Tensor)
        assert t.device.type == "cuda"
        assert torch.all(t == float(i))


def test_cuda_prefetch_handles_dict_items():
    def load(i: int) -> dict:
        return {
            "obs": np.full((4, 8), float(i), dtype="float32"),
            "shot_index": i,
        }

    with CUDAPrefetcher((load(i) for i in range(3)), device="cuda:0") as pf:
        out = list(pf)
    for i, sample in enumerate(out):
        assert sample["obs"].device.type == "cuda"
        # plain ints become 0-d cuda tensors
        assert int(sample["shot_index"]) == i


def test_cuda_prefetch_rejects_cpu_device():
    with pytest.raises(ValueError):
        CUDAPrefetcher((np.zeros(4) for _ in range(2)), device="cpu")
