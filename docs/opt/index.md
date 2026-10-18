# sweepx.opt

PyTorch optimizers tailored for **Full Waveform Inversion (FWI)** in seismic imaging.

## What this library is

`sweepx.opt` is a curated collection of optimization algorithms exposed through
PyTorch's standard `torch.optim.Optimizer` interface, packaged so that they
drop into any PyTorch-based FWI workflow with at most one constructor change.
It pairs naturally with:

- [`sweep`](../solver/index.md) - the seismic forward propagator, which `sweepx`
  installs as a dependency (the library itself is backend-agnostic).
- [`sweepx.loss`](../loss/index.md) - misfit (loss) functions for FWI, in the
  same package.

## What is in it

| Category | Optimizers |
|---|---|
| First-order | Steepest Descent (strong-Wolfe line search), Nonlinear CG (PR+) |
| Quasi-Newton | L-BFGS-FWI, L-BFGS-B |
| Newton-type and trust region | Truncated Newton, Truncated Gauss-Newton, Trust-Region (TTR/ASTR) |
| Adaptive | Adam-FWI (adam / adamw / radam / nadam), Lion |
| Second-order (Kronecker) | Shampoo, SOAP |
| Wrapper | SAM (sharpness-aware) |

They come with the pseudo-Hessian and depth-weighting preconditioners and with
bound, log-velocity, perturbation and DIP-FWI parameterizations.
[The optimizer overview](optimizers/index.md) compares them side by side.

## Get going

- [Installation](getting_started/installation.md)
- [Quick start](getting_started/quickstart.md) - the 5-line drop-in replacement
  for `torch.optim.Adam`
- [Data convention](getting_started/data_convention.md)
