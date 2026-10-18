# Depth weighting

`sweepx.opt.precond.DepthWeighting` multiplies the gradient by
$(z / z_{\max})^{p}$ along the depth axis, compensating for the
surface-illumination bias of seismic acquisitions. With $p > 0$, deep
components of the gradient get amplified relative to shallow ones; $p < 0$
does the opposite.

It composes with any other preconditioner via :class:`sweepx.opt.precond.Compose`,
which chains `__call__`s and forwards `update` kwargs to every wrapped
preconditioner:

```python
import sweepx.opt

precond = sweepx.opt.precond.Compose(
    sweepx.opt.precond.PseudoHessianPrecond(eps=1e-3),
    sweepx.opt.precond.DepthWeighting(depth_axis=-1, p=1.0),
)

opt = sweepx.opt.optim.LBFGS([vp], preconditioner=precond)

# One call updates every preconditioner in the chain:
precond.update(
    [vp],
    source_illum=getattr(solver, "source_illumination", None),
    receiver_illum=getattr(solver, "receiver_illumination", None),
)
```

## API

```python
sweepx.opt.precond.DepthWeighting(
    depth_axis=-1,
    p=1.0,
    z_max_strategy="size",   # "size" (use nz-1) or "unit" (z normalized to [0, 1])
)

sweepx.opt.precond.Compose(*preconds)
```

## Notes

The implementation broadcasts the 1-D weight along the depth axis to the
full parameter shape, so a 2-D `(nx, nz)` model gets weighted uniformly
across `nx` and only varies along `nz`.

`z_max_strategy="size"` (default) uses `z_max = nz - 1`, matching the
common interpretation in geophysics where `z` is a grid index. Switch to
`"unit"` if you want depth normalized to `[0, 1]` for a different
interpretation of `p`.
