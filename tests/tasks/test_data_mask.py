"""Data-mask switch: misfit weighting by an optional per-sample mute mask.
mask off -> bit-identical to legacy; mask on -> excludes window-outside samples."""
import numpy as np
import torch
from types import SimpleNamespace
from sweep_tasks.runner import TaskRunner
from sweep_tasks._helpers.loss import _compute_loss, _loss_sum
from sweep_tasks.schemas import LossSpec


def _pair(shape=(2, 8, 4, 1)):
    g = torch.Generator().manual_seed(0)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def test_mask_off_is_legacy():
    syn, obs = _pair(); spec = LossSpec(kind="mse")
    legacy = _compute_loss(syn, obs, spec).sum()
    assert torch.allclose(_loss_sum(syn, obs, spec, None), legacy)
    assert torch.allclose(_loss_sum(syn, obs, spec, torch.ones_like(syn)), legacy)


def test_mask_excludes_zeroed_region():
    syn, obs = _pair(); spec = LossSpec(kind="mse")
    mask = torch.ones_like(syn); mask[:, 4:, :, :] = 0.0          # mute 2nd half of time
    masked = _loss_sum(syn, obs, spec, mask)
    ref = _compute_loss(syn[:, :4], obs[:, :4], spec).sum()       # loss on kept half only
    assert torch.allclose(masked, ref)


def test_lossspec_field_default_off():
    assert LossSpec().data_mask_path is None
    assert LossSpec(kind="mse", data_mask_path="/tmp/x.npy").data_mask_path == "/tmp/x.npy"


def test_get_data_mask_none_when_unset():
    obs = torch.zeros((2, 8, 4, 1))
    self_ = SimpleNamespace()
    spec = SimpleNamespace(loss=LossSpec(kind="mse"))             # no data_mask_path
    assert TaskRunner._get_data_mask(self_, spec, obs, "cpu") is None


def test_get_data_mask_load_align_cache(tmp_path):
    obs = torch.zeros((2, 8, 4, 1))                               # 4-D obs
    arr = (np.arange(2 * 8 * 4).reshape(2, 8, 4) % 2).astype(np.float32)  # 3-D mask
    p = tmp_path / "m.npy"; np.save(p, arr)
    self_ = SimpleNamespace()
    spec = SimpleNamespace(loss=LossSpec(kind="mse", data_mask_path=str(p)))
    m = TaskRunner._get_data_mask(self_, spec, obs, "cpu")
    assert m.shape == (2, 8, 4, 1)                                # aligned to obs ndim
    assert torch.allclose(m[..., 0], torch.as_tensor(arr))
    # cached (same object on second call)
    assert TaskRunner._get_data_mask(self_, spec, obs, "cpu") is m


def test_get_data_mask_shape_mismatch_raises(tmp_path):
    obs = torch.zeros((2, 8, 4, 1))
    p = tmp_path / "bad.npy"; np.save(p, np.ones((3, 8, 4), np.float32))  # wrong nshots
    self_ = SimpleNamespace()
    spec = SimpleNamespace(loss=LossSpec(kind="mse", data_mask_path=str(p)))
    import pytest
    with pytest.raises(ValueError):
        TaskRunner._get_data_mask(self_, spec, obs, "cpu")
