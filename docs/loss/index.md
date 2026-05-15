# fwiloss

A PyTorch library of misfit (loss) functions for **Full Waveform Inversion (FWI)**.

`fwiloss` collects, behind a single ergonomic API, the loss functions that
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
from fwiloss import L2Loss, HuberLoss, CauchyLoss

syn = torch.randn(2, 1024, 64, 1, requires_grad=True)   # (ns, nt, nr, nc)
obs = torch.randn(2, 1024, 64, 1)

loss = L2Loss()(syn, obs)
loss.backward()
```

## Implemented so far

* L2 — Tarantola (1984)
* L1 — Crase et al. (1990), Brossier et al. (2010)
* Huber / Pseudo-Huber — Guitton & Symes (2003), Charbonnier et al. (1997)
* Hybrid L1/L2 — Bube & Langan (1997)
* Cauchy / Tukey / Geman–McClure — Crase et al. (1990); Aravkin et al. (2012)

A running list of all targeted misfits and their reference is on the
[**References**](references.md) page.

## License

MIT
