"""Local-similarity FWI misfit (Fomel 2007).

The *local similarity* attribute (Fomel 2007) is a windowed normalised
cross-correlation that varies smoothly in time.  For two real signals
:math:`d_s(t), d_o(t)`, define a sliding Gaussian window
:math:`w_\\sigma(t-\\tau)` centred at :math:`\\tau` with half-width
:math:`\\sigma`, and write

.. math::

    \\gamma_\\sigma(\\tau) \\;=\\;
        \\frac{\\sum_t w_\\sigma(t-\\tau)\\,d_s(t)\\,d_o(t)}
             {\\sqrt{\\sum_t w_\\sigma(t-\\tau)\\,d_s(t)^2 \\cdot \\sum_t w_\\sigma(t-\\tau)\\,d_o(t)^2}}.

This is a continuous, smooth analogue of the windowed Pearson
correlation; :math:`\\gamma_\\sigma \\in [-1, 1]` everywhere.

A natural FWI misfit (used e.g. by Zhang, Sirgue & Zhang 2018) is

.. math::

    \\mathcal J_{\\mathrm{LS}}(m) \\;=\\; \\tfrac{1}{2}\\sum_{\\text{trace}}\\sum_\\tau \\bigl(1-\\gamma_\\sigma(\\tau)\\bigr)^2.

It is **amplitude-invariant** at every window and concentrates the
mismatch in time-localised features instead of averaging over the
whole trace as global NCC does.

References
----------
* Fomel, S. (2007). *Local seismic attributes.*  **Geophysics** 72 (3),
  A29-A33.  doi:10.1190/1.2437573
* Zhang, P., Sirgue, L. & Zhang, R. (2018). *Local-similarity-based FWI
  for a time-lapse application.*  80th EAGE Conference & Exhibition.
  doi:10.3997/2214-4609.201801006
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


def _gaussian_kernel(sigma_samples: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian convolution kernel with FWHM-like halfwidth.

    The kernel is truncated at +/- 4 sigma and normalised to unit sum.
    """
    r = max(1, int(4.0 * sigma_samples))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma_samples) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def _smooth(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Conv1d over the last axis with a single Gaussian kernel, replicate pad."""
    nt = x.shape[-1]
    r = kernel.shape[-1] // 2
    xp = torch.nn.functional.pad(x.unsqueeze(1), (r, r), mode="replicate")
    y = torch.nn.functional.conv1d(xp, kernel).squeeze(1)
    return y.view(*x.shape[:-1], nt)


class LocalSimilarityLoss(BaseFWILoss):
    """Windowed-correlation FWI misfit (Fomel 2007; Zhang 2018).

    Parameters
    ----------
    sigma_samples
        Half-width of the Gaussian sliding window, in samples.  Smaller
        ``sigma`` ⇒ more local; larger ⇒ closer to the global NCC misfit.
    eps
        Small constant to stabilise the L2 normalisations.
    """

    def __init__(
        self,
        sigma_samples: float = 8.0,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if sigma_samples <= 0:
            raise ValueError(f"sigma_samples must be > 0, got {sigma_samples}")
        self.sigma_samples = float(sigma_samples)
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _ = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)
        k = _gaussian_kernel(self.sigma_samples, s.device, s.dtype)

        sx = _smooth(s * o, k)              # numerator
        ss = _smooth(s * s, k)
        oo = _smooth(o * o, k)
        denom = torch.sqrt((ss * oo).clamp_min(self.eps))
        gamma = sx / denom                  # shape (N, nt) in [-1, 1] approx

        # Energy weight: only count samples where BOTH signals carry energy.
        # Without this, silent-trace boundaries contribute spurious (1-0)^2 =
        # 1 to the loss because gamma = 0/sqrt(eps) -> 0 there.
        energy = ss * oo
        peak = energy.amax(dim=-1, keepdim=True).clamp_min(self.eps)
        w = energy / peak                   # in [0, 1] per sample

        pw = 0.5 * w * (1.0 - gamma) ** 2
        if self.reduction == "mean":
            return pw.mean()
        if self.reduction == "sum":
            return pw.sum()
        return pw


def local_similarity_loss(
    syn,
    obs,
    sigma_samples: float = 8.0,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return LocalSimilarityLoss(
        sigma_samples=sigma_samples, eps=eps, reduction=reduction
    )(syn, obs)


__all__ = ["LocalSimilarityLoss", "local_similarity_loss"]
