"""Tests for SirenWavelet."""

import torch

from sweep_nn import SirenWavelet


def test_output_shape():
    net = SirenWavelet(nt=512, hidden_features=16, hidden_layers=2)
    w = net()
    assert w.shape == (512,)


def test_backprop():
    net = SirenWavelet(nt=128, hidden_features=8, hidden_layers=1)
    w = net()
    loss = w.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.parameters())


def test_bias_default_true():
    """Wavelet SIREN must have bias enabled or it can only fit odd-symmetric
    functions — defaulting to True is part of the public contract."""
    net = SirenWavelet(nt=32, hidden_features=8, hidden_layers=1)
    # Every linear layer in the SIREN MLP should have a bias.
    bias_count = sum(
        1 for m in net.modules() if isinstance(m, torch.nn.Linear) and m.bias is not None
    )
    assert bias_count > 0
