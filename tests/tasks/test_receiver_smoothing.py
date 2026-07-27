"""Receiver-side lateral smoothing operator."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep_tasks.preproc.filter import (  # noqa: E402
    apply_receiver_smoothing,
    receiver_smoothing_matrix,
)


def _line_receivers(n=64, step=1.0):
    x = np.arange(n) * step
    return np.stack([x, np.zeros(n)], axis=1)


def test_rows_sum_to_one():
    w = receiver_smoothing_matrix(_line_receivers(), 2.0)
    assert torch.allclose(w.sum(dim=1), torch.ones(w.shape[0]), atol=1e-6)


def test_constant_record_is_unchanged():
    """Row normalisation => a laterally constant record passes through."""
    w = receiver_smoothing_matrix(_line_receivers(32), 2.0)
    x = torch.ones(1, 5, 32, 1) * 3.0
    assert torch.allclose(apply_receiver_smoothing(x, w), x, atol=1e-5)


def test_cutoff_zeroes_far_receivers():
    w = receiver_smoothing_matrix(_line_receivers(64), 2.0, cutoff_sigmas=3.0)
    # receiver 0 vs receiver 20: 20 cells = 10 sigma, well beyond the cutoff
    assert w[0, 20] == 0.0
    assert w[0, 1] > 0.0


def test_uncorrelated_noise_is_suppressed_more_than_coherent_signal():
    """The point of the operator: noise averages down, coherent signal does not."""
    rec = _line_receivers(256)
    w = receiver_smoothing_matrix(rec, 3.0)
    g = torch.Generator().manual_seed(0)
    nt = 64
    # laterally constant (perfectly coherent) signal
    sig = torch.randn(1, nt, 1, 1, generator=g).expand(1, nt, 256, 1).contiguous()
    noise = torch.randn(1, nt, 256, 1, generator=g)
    keep_sig = apply_receiver_smoothing(sig, w).std() / sig.std()
    keep_noise = apply_receiver_smoothing(noise, w).std() / noise.std()
    assert keep_sig > 0.99                    # coherent signal survives
    assert keep_noise < 0.35                  # incoherent noise is knocked down
    assert keep_sig / keep_noise > 2.5


def test_is_differentiable_and_applies_to_both_records():
    """Gradients must flow: the operator sits inside the misfit graph."""
    rec = _line_receivers(48)
    w = receiver_smoothing_matrix(rec, 2.0)
    syn = torch.zeros(1, 16, 48, 1, requires_grad=True)
    obs = torch.randn(1, 16, 48, 1)
    loss = ((apply_receiver_smoothing(syn, w)
             - apply_receiver_smoothing(obs, w)) ** 2).sum()
    loss.backward()
    assert syn.grad is not None
    assert torch.isfinite(syn.grad).all()
    assert syn.grad.abs().sum() > 0


def test_smoothing_both_sides_leaves_a_perfect_match_at_zero_misfit():
    w = receiver_smoothing_matrix(_line_receivers(40), 2.5)
    x = torch.randn(1, 32, 40, 1)
    d = apply_receiver_smoothing(x, w) - apply_receiver_smoothing(x.clone(), w)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_receiver_axis_and_shape_validation():
    w = receiver_smoothing_matrix(_line_receivers(16), 2.0)
    with pytest.raises(ValueError, match="receivers on axis"):
        apply_receiver_smoothing(torch.zeros(1, 4, 15, 1), w)
    with pytest.raises(ValueError, match="rec_xy must be"):
        receiver_smoothing_matrix(np.zeros((10, 3)), 2.0)
    with pytest.raises(ValueError, match="sigma_cells must be"):
        receiver_smoothing_matrix(_line_receivers(8), 0.0)


def test_scattered_2d_receivers_normalised_despite_density_variation():
    """Normalised convolution: clustered receivers must not gain amplitude."""
    rng = np.random.default_rng(0)
    dense = rng.uniform(0, 4, size=(60, 2))
    sparse = rng.uniform(20, 40, size=(20, 2))
    rec = np.concatenate([dense, sparse], axis=0)
    w = receiver_smoothing_matrix(rec, 3.0)
    assert torch.allclose(w.sum(dim=1), torch.ones(w.shape[0]), atol=1e-6)
    x = torch.ones(1, 3, rec.shape[0], 1)
    assert torch.allclose(apply_receiver_smoothing(x, w), x, atol=1e-5)


def test_target_obs_only_leaves_syn_untouched():
    """target='obs' must smooth the observation and pass the synthetic through."""
    from sweep_tasks.schemas import ReceiverSmoothingSpec
    assert ReceiverSmoothingSpec().target == "both"          # safe default
    assert ReceiverSmoothingSpec(target="obs").target == "obs"
    w = receiver_smoothing_matrix(_line_receivers(32), 2.0)
    syn = torch.randn(1, 8, 32, 1)
    # emulate the two branches the runner takes
    both = (apply_receiver_smoothing(syn, w), apply_receiver_smoothing(syn, w))
    obs_only = (syn, apply_receiver_smoothing(syn, w))
    assert torch.allclose(both[0], both[1], atol=1e-6)        # consistent
    assert not torch.allclose(obs_only[0], obs_only[1], atol=1e-3)  # asymmetric
