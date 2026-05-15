# Quick start

```python
import torch
from fwiloss import L2Loss, L1Loss, HuberLoss

ns, nt, nr, nc = 2, 1024, 64, 1
syn = torch.randn(ns, nt, nr, nc, requires_grad=True)
obs = torch.randn(ns, nt, nr, nc)

# Classical least-squares
mse = L2Loss()(syn, obs)

# Robust alternatives
l1 = L1Loss()(syn, obs)
hu = HuberLoss(delta=0.5)(syn, obs)

mse.backward()
```

## Choosing a misfit

A rough decision tree:

* High SNR, good initial model → `L2Loss`.
* Outliers / impulsive noise → `L1Loss`, `HuberLoss`, `CauchyLoss`,
  `TukeyLoss`.
* Strong cycle-skipping → envelope, instantaneous-phase, optimal-transport
  or AWI families (coming soon).
* Frequency-domain inversion → Pratt-style frequency-domain L2 or the
  Shin-Min log-amplitude/phase split (coming soon).

See the per-loss pages on the left for the precise formulas and tested
parameter ranges, and the [Report](../report.md) for the full bibliography.
