# sweepx.opt

PyTorch optimizers tailored for **Full Waveform Inversion (FWI)** in seismic imaging.

## What this library is

`sweepx.opt` is a curated collection of optimization algorithms exposed through
PyTorch's standard `torch.optim.Optimizer` interface, packaged so that they
drop into any PyTorch-based FWI workflow with at most one constructor change.
It pairs naturally with:

- [`sweep`](https://github.com/DeepWave-KAUST/sweep) - the seismic forward
  propagator (the library itself is backend-agnostic).
- [`sweep-loss`](https://github.com/DeepWave-KAUST/sweep-loss) - misfit (loss)
  functions for FWI.

All three are part of [`sweepx`](https://github.com/DeepWave-KAUST/sweepx).

## What ships when

The library is built out commit-by-commit, with one optimizer per commit and a
mkdocs page for each.

| Tier | Adds | Commits |
|---|---|---|
| Tier-1 | Steepest Descent (Wolfe) - Pseudo-Hessian preconditioner - Nonlinear CG (PR+) - L-BFGS-FWI | 6 |
| Tier-2 | HVP/CG infra - Truncated Newton - Truncated Gauss-Newton - Trust-Region (ASTR/TTR) - L-BFGS-B - parameterizations + depth weighting | 7 |
| Tier-3 | Adam-FWI (adam/adamw/radam/nadam) - Lion - SAM - Shampoo - SOAP - DIP-FWI parameterization | 7 |

See [the optimizer overview](optimizers/index.md) for the comparison matrix
that grows with each Tier.

## Get going

- [Installation](getting_started/installation.md)
- [Quick start](getting_started/quickstart.md) - the 5-line drop-in replacement
  for `torch.optim.Adam`
- [Data convention](getting_started/data_convention.md)
