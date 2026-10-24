# Depth weighting

`sweepx.opt.precond.DepthWeighting` multiplies the gradient by
$(z / z_{\max})^{p}$ along the depth axis, compensating for the
surface-illumination bias of seismic acquisitions. With $p > 0$, deep
components of the gradient get amplified relative to shallow ones; $p < 0$
does the opposite.

It composes with any other preconditioner via `sweepx.opt.precond.Compose`,
which chains `__call__`s and forwards `update` kwargs to every wrapped
preconditioner:

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
precond = sweepx.opt.precond.Compose(
    sweepx.opt.precond.PseudoHessianPrecond(eps=1e-5),
    sweepx.opt.precond.DepthWeighting(depth_axis=0, p=1.0),   # (nz, nx): depth is axis 0
)
solver.compute_illumination = True   # impl='c': backward() fills the illumination maps

opt = sweepx.opt.optim.LBFGS([vp], preconditioner=precond, initial_step=20.0)

def closure():
    opt.zero_grad()
    loss = (solver(wavelet, sources, receivers, models=[vp]) - obs).pow(2).sum()
    loss.backward()
    # One call updates every preconditioner in the chain:
    S, R = solver.source_illumination, solver.receiver_illumination
    precond.update([vp], source_illum=S / S.max(), receiver_illum=R / R.max())
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

## API

```python
sweepx.opt.precond.DepthWeighting(
    depth_axis=0,            # sweep models: (nz, nx) and (nz, ny, nx)
    p=1.0,
    z_max_strategy="size",   # "size" (use nz-1) or "unit" (z normalized to [0, 1])
)

sweepx.opt.precond.Compose(*preconds)
```

## Notes

The implementation broadcasts the 1-D weight along the depth axis to the
full parameter shape, so a 2-D `(nz, nx)` model gets weighted uniformly
across `nx` and only varies along `nz`. sweep stores depth first in 2-D and
3-D (`(nz, nx)`, `(nz, ny, nx)`), so the default `depth_axis=0` fits both;
use `depth_axis=1` for a batched `(B, nz, ...)` tensor. (sweepx 0.4.0 and
earlier defaulted to `-1`, which weights along `x` on sweep models.)

`z_max_strategy="size"` (default) uses `z_max = nz - 1`, matching the
common interpretation in geophysics where `z` is a grid index. Switch to
`"unit"` if you want depth normalized to `[0, 1]` for a different
interpretation of `p`.
