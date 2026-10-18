# sweepx.loss

A PyTorch library of misfit (loss) functions for **Full Waveform Inversion (FWI)**.

`sweepx.loss` collects, behind a single ergonomic API, the loss functions that
have been proposed in the geophysical FWI literature — from classical
least-squares to optimal-transport / adaptive matching-filter / envelope /
phase variants — and exposes every one of them as a `torch.nn.Module` so
they drop directly into any PyTorch-based FWI workflow (e.g. the
[`sweep`](https://github.com/DeepWave-KAUST/sweep) propagator).

## Why a dedicated package?

Generic ML losses (cross-entropy, focal, contrastive, …) are **not** what
the seismic-inversion community usually means by "loss".  The misfits
relevant here come from the FWI / seismic-tomography papers of the last
40 years, and they need:

* a canonical 4-D data layout `(nshots, nt, nreceivers, nchannel)`,
* gradient flow through PyTorch autograd into the model parameters,
* a small zoo of physics-aware operations (Hilbert envelope, CDF transport,
  Wiener filter, cross-correlation, &c.).

See the [**Report**](report.md) for the complete catalogue with formulas
and citations (every reference carries a DOI).

## Quick taste

```python
import torch
from sweepx.loss import L2Loss, HuberLoss, CauchyLoss

syn = torch.randn(2, 1024, 64, 1, requires_grad=True)   # (ns, nt, nr, nc)
obs = torch.randn(2, 1024, 64, 1)

loss = L2Loss()(syn, obs)
loss.backward()
```

## What is implemented

Thirty-three misfits, from least squares and robust norms to envelope, phase,
matching-filter and optimal-transport misfits. The pages under
[**Loss functions**](losses/index.md) give each one's formula, its reference with
a DOI and the tests that check it; the [**References**](references.md) page lists
every source paper.

## License

MIT
