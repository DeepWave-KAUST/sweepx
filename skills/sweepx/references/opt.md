# Optimizers, bounds, preconditioners

## Contents

- [In a task YAML](#in-a-task-yaml)
- [sweepx.opt in Python](#sweepxopt-in-python) (runnable)
- [Step sizes](#step-sizes)

## In a task YAML

The runner builds `torch.optim` Adam / SGD, and for `kind: lbfgs` the
`sweepx.opt` L-BFGS (strong-Wolfe line search, bounded by `model_bounds` when
one model is inverted). The rest of `sweepx.opt` (L-BFGS-B, NCG, Newton, SOAP,
...) is **not** selectable from a YAML.

| block | options | notes |
|---|---|---|
| `optimizer: {kind: adam, lr, eps: 1e-22, betas: [0.9, 0.999]}` | `lr` scalar or `{vp: 20, vs: 5}` per inverted model | default choice; grid `lr` ≈ m/s per step (5–50) |
| `optimizer: {kind: sgd, lr, momentum: 0.9, nesterov, weight_decay}` | per-model `lr` dict allowed | `lr` multiplies the raw gradient (tiny for velocities) |
| `optimizer: {kind: lbfgs, lr: 1.0, initial_step: auto, max_iter: 1, history_size: 10}` | single `lr` (multiplies the quasi-Newton step) | `initial_step` = first step in model units (`auto` = 1% of mean \|model\|; set it for LSRTM, whose image starts at 0); `max_iter` iterations per epoch; needs `batchsize` ≥ shot count; refused under torchrun (`--nproc-per-node > 1`). Since 0.4.1; before, `torch.optim.LBFGS` took steps in gradient units |
| `scheduler: {kind: constant \| step \| exp \| cosine}` | `step_size`/`gamma`, `gamma`, `t_max`/`eta_min` | one schedule over the global epoch of all stages; stage `lr_scale` sets each stage's base |
| `model_bounds: {vp: {min, max, enabled}}` | per model name | in-place clamp after every step |
| `illumination_precondition: {enabled, epsilon, exponent: 0.5, relative_epsilon}` | | divides the gradient by `(S*R + eps)^exponent`; needs `impl: c` illumination. `relative_epsilon` (e.g. 1e-3) is safer than an absolute `epsilon` |
| `grad_smooth: {sigma_z_cells, sigma_x_cells, sigma_y_cells, every, mask_above_row, ...}` | | Gaussian-smooths the vp gradient (grid mode) |
| `smooth_regularization: {weight, order, x/y/z_weight, velocity_scale_m_s}` | | adds `weight * TV(vp)` to the loss |

Under `reparam:` the optimizer kind/eps/betas apply but `lr` comes from
`reparam.lr`. Under frequency selection only Adam at torch defaults
(eps 1e-8) is built, from `optimizer.lr` (or `reparam.lr`) × the stage scale.

## sweepx.opt in Python

Every class subclasses `torch.optim.Optimizer`.

| class | `step()` takes | notes |
|---|---|---|
| `AdamFWI(params, lr, variant="adam", eps=1e-22)` | optional closure | `lr` ≈ change per step in parameter units |
| `Lion`, `Shampoo`, `SOAP` | optional closure | adaptive; Shampoo's per-cell change is smaller than `lr` |
| `SteepestDescent` | closure only with `line_search` | `lr` multiplies the raw gradient |
| `NCG`, `LBFGS`, `LBFGSB` | closure | set `initial_step=` (parameter units) or the first line search barely moves |
| `TruncatedNewton`, `TrustRegion` | `loss_fn` (no `backward()` inside) | need a twice-differentiable solver: `impl='eager'`, no `torch.compile` |
| `TruncatedGaussNewton` | `residual_fn` | same |
| `SAM(base)` | closure | base must not need a closure |

All but SAM accept `bounds=(lo, hi)` (iterates and line-search trials are
projected) and the first-order / quasi-Newton / Newton ones accept
`preconditioner=`.

| module | classes |
|---|---|
| `sweepx.opt.precond` | `PseudoHessianPrecond(eps=1e-3, normalize=True)` (refresh with `.update(params, source_illum=, receiver_illum=)` after each `backward()`; sweep fills `solver.source_illumination` / `receiver_illumination` only with `impl='c'` and `solver.compute_illumination = True`), `DepthWeighting(depth_axis=0, p=1.0)`, `Compose(*preconds)`, `IdentityPrecond` |
| `sweepx.opt.parameterization` | `SigmoidBounds(vmin, vmax)`, `LogVelocity`, `PerturbationFromBackground`, `DIPParameterization`; `latent = p.inverse(model)`, solver sees `p.forward(latent)` |
| `sweepx.opt.linalg`, `sweepx.opt.linesearch` | CG, Hessian-vector products, strong-Wolfe search |

A small runnable loop (needs only `sweep` and `sweepx`; eager solver so it
also runs on CPU):

```python
import numpy as np, torch
import sweepx.opt
from sweepx.opt.parameterization import SigmoidBounds
from sweep.equations import Acoustic
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz, nx, nt, dt = 48, 64, 400, 1e-3
solver = PropTorch(Acoustic(device=dev), shape=(nz, nx), dh=10.0, dt=dt, device=dev, impl="eager",
                   eager_options=EagerOptions(use_compile=False))   # skip torch.compile on tiny runs
wavelet = ricker(np.arange(nt) * dt - 0.1, f=12.0).astype(np.float32)
sources = np.array([[16, 2], [48, 2]], dtype=np.int64)                 # (nshots, 2): x, z
receivers = np.stack([np.c_[np.arange(0, nx, 2), np.full(nx // 2, 2)]] * 2).astype(np.int64)

vp_true = torch.full((nz, nx), 2000.0, device=dev); vp_true[24:34, 20:44] = 2300.0
with torch.no_grad():
    obs = solver(wavelet, sources, receivers, models=[vp_true])        # (2, nt, 32, 1)

def misfit(vp):
    return (solver(wavelet, sources, receivers, models=[vp]) - obs).pow(2).sum()

# 1) Adam-FWI: lr in m/s per step, hard bounds
vp = torch.full((nz, nx), 2000.0, device=dev, requires_grad=True)
opt = sweepx.opt.optim.AdamFWI([vp], lr=10.0, bounds=(1500.0, 3000.0))
for _ in range(3):
    opt.zero_grad(); loss = misfit(vp); loss.backward(); opt.step()
print("adam", loss.item())

# 2) L-BFGS with a closure, depth-weighted gradient, first trial step 20 m/s
vp = torch.full((nz, nx), 2000.0, device=dev, requires_grad=True)
opt = sweepx.opt.optim.LBFGS([vp], bounds=(1500.0, 3000.0), initial_step=20.0,
                             preconditioner=sweepx.opt.precond.DepthWeighting(depth_axis=0, p=1.0))
def closure():
    opt.zero_grad(); l = misfit(vp); l.backward(); return l
for _ in range(2):
    loss = opt.step(closure)
print("lbfgs", float(loss.detach()))

# 3) optimise a latent; the solver sees a model inside (1500, 3000) by construction
param = SigmoidBounds(vmin=1500.0, vmax=3000.0)
latent = param.inverse(torch.full((nz, nx), 2000.0, device=dev)).detach().requires_grad_(True)
opt = sweepx.opt.optim.AdamFWI([latent], lr=0.05)                     # latent units, not m/s
for _ in range(2):
    opt.zero_grad(); loss = misfit(param.forward(latent)); loss.backward(); opt.step()
print("sigmoid", loss.item(), float(param.forward(latent).max()))
```

## Step sizes

The gradient of a waveform misfit with respect to velocity in m/s is tiny
(~1e-4..1e-3 on small problems), so:

- **Adam, AdamFWI, Lion, SOAP**: `lr` ≈ change per step in parameter units —
  pick 5–50 for m/s grids; under a latent parameterization use latent units.
- **SGD / SteepestDescent without line search**: `lr` multiplies the raw
  gradient; it needs a huge value and must keep the model under the CFL limit.
- **Line-search methods**: give `initial_step` (parameter units) or every
  early step moves the model by almost nothing (a `RuntimeWarning` says so).
- sweep does not check CFL inside your own loop: a trial step past the
  stability limit returns NaN data, not an error. Bound the model.
