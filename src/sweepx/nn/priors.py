"""Learned-prior helpers.

The intended use is for *perceptual* / feature-based losses: wrap a frozen
feature extractor (e.g. a CNN pretrained on velocity-model patches) and
compute distances in feature space inside your loss function.

The wrapper deliberately freezes the underlying module's parameters so
gradients flow to whatever produced the input, never to the prior itself.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LearnedPrior(nn.Module):
    """Wrap a feature extractor; freeze its parameters; expose ``features(x)``.

    Parameters
    ----------
    backbone
        An ``nn.Module`` whose ``forward(x)`` produces either a single
        feature tensor or a tuple/list of multi-scale tensors.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "LearnedPrior":  # type: ignore[override]
        # keep backbone in eval regardless
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        out = self.backbone(x)
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(out)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.features(x)


def perceptual_distance(
    prior: LearnedPrior,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    weights: Iterable[float] | None = None,
    p: int = 2,
) -> torch.Tensor:
    """L_p distance averaged across the prior's feature maps.

    Parameters
    ----------
    prior
        A :class:`LearnedPrior`.
    a, b
        Inputs in the same shape the backbone expects.
    weights
        Per-scale weights (same length as the backbone's feature list).
    p
        Distance order. ``2`` for L2, ``1`` for L1.
    """
    fa = prior.features(a)
    fb = prior.features(b)
    if len(fa) != len(fb):
        raise RuntimeError(
            f"prior produced {len(fa)} feature maps for `a` and {len(fb)} for `b`"
        )
    if weights is None:
        ws = [1.0] * len(fa)
    else:
        ws = list(weights)
        if len(ws) != len(fa):
            raise ValueError(f"expected {len(fa)} weights; got {len(ws)}")
    total = a.new_zeros(())
    for w, x, y in zip(ws, fa, fb):
        total = total + w * (x - y).abs().pow(p).mean()
    return total


__all__ = ["LearnedPrior", "perceptual_distance"]
