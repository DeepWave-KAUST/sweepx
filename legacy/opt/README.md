# sweep-opt

PyTorch optimizers tailored for **Full Waveform Inversion (FWI)** in seismic imaging.

`sweep-opt` collects optimization algorithms that have proven useful for FWI - from
classical line-search and quasi-Newton methods to Newton-type, trust-region, and modern
adaptive / second-order machine-learning optimizers - and exposes every one of them
through the standard `torch.optim.Optimizer` interface so they drop into any PyTorch
FWI workflow (e.g. the [`sweep`](https://github.com/DeepWave-KAUST/sweep) propagator,
or any user-built autograd FWI loop).

## Why a separate library

PyTorch ships only a handful of generic optimizers. FWI practitioners typically need:
- bound-constrained quasi-Newton (L-BFGS / L-BFGS-B with velocity bounds)
- preconditioning by the pseudo-Hessian / depth weighting
- Newton-type and trust-region methods that talk to the Hessian via Hessian-vector products
- modern stochastic optimizers tuned for source-encoded mini-batches
- model reparameterizations (sigmoid bounds, log-velocity, deep image prior)

These either don't exist in `torch.optim` or live in scattered research codes.
`sweep-opt` provides a curated, tested, documented collection.

## Quick start

```python
import torch
import sweep_opt

vp = torch.randn(64, 128, requires_grad=True)

# Drop-in replacement for torch.optim.Adam (no closure required):
opt = sweep_opt.optim.SteepestDescent([vp], lr=1e-3)
opt.zero_grad(); loss = forward(vp); loss.backward(); opt.step()

# L-BFGS with velocity bounds + pseudo-Hessian preconditioning (closure required):
precond = sweep_opt.precond.PseudoHessianPrecond(eps=1e-3)
opt = sweep_opt.optim.LBFGS([vp], history_size=10,
                            bounds=(1500.0, 5500.0), preconditioner=precond)
opt.step(closure)
```

## Installation

```bash
pip install -e .                 # core (torch, numpy)
pip install -e .[dev]            # + pytest, ruff, black, mkdocs
pip install -e .[sweep]          # + sweep (needed only by examples/marmousi/)
```

## Documentation

The repo ships an [MkDocs](https://www.mkdocs.org/) +
[Material](https://squidfunk.github.io/mkdocs-material/) site:

```bash
pip install -e .[docs]
mkdocs serve   # http://127.0.0.1:8000
```

Each algorithm has its own page under `docs/optimizers/`,
`docs/preconditioners/`, or `docs/parameterization/`. Convergence comparisons on
the 2-D acoustic Marmousi benchmark live in [`docs/report.md`](docs/report.md).

## Companion libraries

- [`sweep`](https://github.com/DeepWave-KAUST/sweep) - the seismic forward propagator
  this library benchmarks against. Consumed via the optional `[sweep]` extra.
- [`fwiloss`](https://github.com/DeepWave-KAUST/sweep-loss) - misfit (loss) functions for FWI
  (L2, Huber, envelope, Wasserstein, AWI, ...). Pair with any sweep-opt optimizer.
