# Quick start

`sweepx.opt` exposes every optimizer through PyTorch's standard
`torch.optim.Optimizer` interface, so the entire library can be adopted by
swapping a single line.

## Setup used on these pages

The code on the `sweepx.opt` pages runs after this block: the forward model
of [Getting started](../../getting-started.md), its data for a model with a
faster box as `obs`, and a constant 2000 m/s starting model.

```python
import numpy as np, torch
import sweepx.opt
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

dev    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
solver = PropTorch(Acoustic(device=dev), shape=(96, 128), dh=10.0, dt=1e-3, device=dev)

wavelet   = ricker(np.arange(1200) * 1e-3 - 0.12, f=10.0).astype(np.float32)
sources   = np.array([[64, 2]], dtype=np.int64)
receivers = np.array([[[ix, 4] for ix in range(0, 128, 2)]], dtype=np.int64)

vp_true = torch.full((96, 128), 2000.0, device=dev)   # (nz, nx)
vp_true[48:58, 32:96] = 2300.0
with torch.no_grad():
    obs = solver(wavelet, sources, receivers, models=[vp_true])

init_vp = torch.full((96, 128), 2000.0, device=dev)   # starting model
inv_vp  = init_vp.clone().requires_grad_(True)
cfg     = {"lr": 10.0, "epochs": 5}
```

The time step is 1 ms instead of the 2 ms of Getting started, so that the
solver stays stable up to 5500 m/s, the upper velocity bound of the examples:
with `dh=10.0` and `dt=2e-3` it is stable at 3000 m/s and returns NaN at
3300 m/s. sweep does not check the stability (CFL) condition, so a model or a
trial step beyond it gives NaN data rather than an error.

## The drop-in replacement

Any PyTorch FWI loop already looks like this:

```python
import torch
optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

for epoch in range(cfg["epochs"]):
    optimizer.zero_grad()
    syn  = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    optimizer.step()
```

For an optimizer whose `step()` works without a closure (Adam-FWI, Lion,
Shampoo, SOAP, Steepest Descent without a line search), change only the
constructor:

```python
import sweepx.opt
optimizer = sweepx.opt.optim.AdamFWI([inv_vp], lr=cfg["lr"])
```

`AdamFWI` keeps the meaning of `lr` (about the change per step, m/s here).
Steepest Descent's `lr` multiplies the raw gradient instead; see
[step sizes](#step-sizes-are-in-parameter-units) below.

## Optimizers that need a closure

L-BFGS, L-BFGS-B, NCG, Steepest Descent with `line_search="strong_wolfe"`
and SAM evaluate the loss more than once per step. Wrap the loss block in a
function and pass it to `step()`:

```python
optimizer = sweepx.opt.optim.LBFGS([inv_vp], initial_step=20.0)   # first trial: 20 m/s

def closure():
    optimizer.zero_grad()
    syn  = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    return loss

for epoch in range(cfg["epochs"]):
    optimizer.step(closure)
```

The Newton-type optimizers differ: `TruncatedNewton` and `TrustRegion` take a
`loss_fn` that returns the loss **without** calling `backward()`,
`TruncatedGaussNewton` takes a `residual_fn` that returns the residual, and
all three need a solver that can be differentiated twice (sweep's
`impl='eager'` without `torch.compile`). See
[Truncated Newton](../optimizers/newton.md),
[Truncated Gauss-Newton](../optimizers/gauss_newton.md) and
[Trust-Region](../optimizers/trust_region.md).

## Step sizes are in parameter units

The gradient of this misfit with respect to a velocity in m/s is small: at
`init_vp` its largest entry is about `7e-4`, while one float32 step at
2000 m/s is `1.2e-4` m/s. A step parameter that multiplies the raw gradient
needs a matching scale:

- **Adam-FWI, Lion, SOAP** normalize the update per coordinate (SOAP in a
  rotated basis); `lr` is roughly the change per step in parameter units
  (m/s here). **Shampoo** normalizes the update as a matrix (its first step
  is `lr` times an orthogonal factor), so per-cell changes are smaller than
  `lr`.
- **Steepest Descent** without a line search steps `lr * P g`, so `lr` is in
  (m/s) / (gradient units), and it must also stay below the stability limit
  set by the stiffest part of the model; see
  [Steepest Descent](../optimizers/steepest_descent.md#drop-in-usage).
- **L-BFGS, L-BFGS-B, NCG and Steepest Descent with a line search** start
  each line search at `alpha_init * d`, with `d` built from the raw gradient.
  `initial_step=` sizes the first trial in parameter units instead. Without
  it, on this setup every NCG or Steepest Descent step moves the model by
  less than 0.01 m/s (five steps change the misfit by less than 0.1 %) and
  L-BFGS wastes its first step the same way; each such search emits a
  `RuntimeWarning` (it stopped at `alpha_max` or left the parameters
  unchanged).
- **TrustRegion** (`initial_radius`, `max_radius`) and **SAM** (`rho`) take
  Euclidean norms over all parameters, in parameter units.
- **Truncated Newton / Gauss-Newton** solve for the step itself, so their
  `alpha_init=1` is already in parameter units.

## With bounds and a preconditioner

`bounds=(lo, hi)` keeps every iterate, and every line-search trial, inside
the box. The [pseudo-Hessian preconditioner](../preconditioners/pseudo_hessian.md)
needs illumination maps, which sweep's native backend (`impl='c'`, the
default on a GPU) fills during `backward()` once `compute_illumination` is
on. Refresh the preconditioner in the closure, right after `backward()`, so
every step direction uses the maps of the model it starts from:

```python
import sweepx.opt

precond = sweepx.opt.precond.PseudoHessianPrecond(eps=1e-5)
solver.compute_illumination = True

optimizer = sweepx.opt.optim.LBFGS(
    [inv_vp],
    history_size=10,
    bounds=(1500.0, 5500.0),         # m/s velocity bounds
    preconditioner=precond,
    initial_step=20.0,
    restart_every=20,
)

def closure():
    optimizer.zero_grad()
    syn  = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    S, R = solver.source_illumination, solver.receiver_illumination
    precond.update([inv_vp], source_illum=S / S.max(), receiver_illum=R / R.max())
    return loss

for epoch in range(cfg["epochs"]):
    optimizer.step(closure)
```

The [optimizer overview](../optimizers/index.md) lists every algorithm and
what its `step()` takes.
