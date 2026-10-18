# Preconditioners

Preconditioners in `sweep-opt` are objects that pre-multiply a gradient
before an optimizer's `step()` consumes it. They share a single contract:

```python
class Preconditioner:
    def update(self, params, *, source_illum=None, receiver_illum=None) -> None:
        """Refresh internal state. Called once per outer iteration."""

    def __call__(self, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """Return the preconditioned gradient."""
```

This means any preconditioner composes with any optimizer:

```python
optimizer = sweep_opt.optim.LBFGS(
    [inv_vp],
    preconditioner=sweep_opt.precond.PseudoHessianPrecond(eps=1e-3),
)
```

Preconditioners can also be composed with each other (e.g. pseudo-Hessian
multiplied by depth weighting); see the per-algorithm pages.

## Available preconditioners

| Name | Class | Tier | Notes |
|---|---|---|---|
| [Pseudo-Hessian](pseudo_hessian.md) | `PseudoHessianPrecond` | 1 | Plessix-Mulder diagonal; needs source / receiver illumination. |
| [Depth weighting](depth_weighting.md) | `DepthWeighting` | 2 | `(z / z_max)^p` mask along the depth axis; composable with PH. |
