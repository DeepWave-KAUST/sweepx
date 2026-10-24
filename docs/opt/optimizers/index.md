# Optimizers

This page lists every optimizer shipped by `sweepx.opt`, grouped by category.

| Algorithm | Class | Category | `step()` takes | Bounds | Preconditioner |
|---|---|---|---|---|---|
| [Steepest Descent](steepest_descent.md) | `SteepestDescent` | First-order | closure only with `line_search` | yes | yes |
| [Nonlinear CG (PR+)](ncg.md) | `NCG` | First-order | closure | yes | yes |
| [L-BFGS-FWI](lbfgs.md) | `LBFGS` | Quasi-Newton | closure | yes (projected) | yes (as `H_0`) |
| [Truncated Newton (HF)](newton.md) | `TruncatedNewton` | Newton-type | `loss_fn`, no `backward` | yes | yes (PCG inner) |
| [Truncated Gauss-Newton](gauss_newton.md) | `TruncatedGaussNewton` | Newton-type | `residual_fn`, no `backward` | yes | yes (PCG inner) |
| [Trust-Region (TTR/ASTR)](trust_region.md) | `TrustRegion` | Trust-region | `loss_fn`, no `backward` | yes | yes (PCG inner, TTR) |
| [L-BFGS-B](lbfgsb.md) | `LBFGSB` | Quasi-Newton | closure | yes (active-set) | yes (as `H_0`) |
| [Adam-FWI](adam_fwi.md) | `AdamFWI` | Adaptive | optional closure | yes | n/a (per-param adaptive) |
| [Lion](lion.md) | `Lion` | Adaptive (sign-momentum) | optional closure | yes | n/a |
| [SAM](sam.md) | `SAM` | Wrapper (sharpness-aware) | closure | from the base | from the base |
| [Shampoo](shampoo.md) | `Shampoo` | Second-order (Kronecker) | optional closure | yes | n/a (Kronecker-factored built in) |
| [SOAP](soap.md) | `SOAP` | Second-order (Shampoo + Adam) | optional closure | yes | n/a |

See each algorithm's own page for the API, theory, and references. The code
on these pages runs after the
[setup block](../getting_started/quickstart.md#setup-used-on-these-pages)
(`solver`, `wavelet`, `sources`, `receivers`, `obs`, `init_vp`, `inv_vp`,
`cfg`).

## Conventions

Every optimizer in `sweepx.opt`:

1. Subclasses `torch.optim.Optimizer`, so it works with any code that already
   uses `torch.optim.Adam`/`SGD`/`LBFGS`.
2. Implements `step(closure=None)` with the same closure semantics as
   `torch.optim.LBFGS`: closure must `zero_grad()`, recompute the loss, call
   `backward()`, and return the loss. The exceptions are `TruncatedNewton`
   and `TrustRegion`, whose `step()` requires a `loss_fn`, and
   `TruncatedGaussNewton`, whose `step()` requires a `residual_fn`; neither
   calls `backward()`, and the loss must be twice differentiable (sweep:
   `impl='eager'` without `torch.compile`). SAM's base optimizer must not
   need a closure.
3. Accepts an optional `bounds=(lo, hi)` argument (all but SAM, which uses
   its base's); if provided, parameters are projected to the box after each
   update, and line-search and trust-region trial points are projected
   before they are evaluated.
4. The first-order, quasi-Newton, Newton-type and trust-region optimizers
   accept an optional `preconditioner=` that pre-multiplies gradients before
   the step. See [Preconditioners](../preconditioners/index.md).
5. Step-size parameters are either in parameter units (`lr` of the adaptive
   optimizers, `initial_step`, trust radii, SAM's `rho`) or multiply the raw
   gradient (`lr` of Steepest Descent, `alpha_init` / `alpha_max` of the
   first-order and quasi-Newton line searches); see
   [step sizes](../getting_started/quickstart.md#step-sizes-are-in-parameter-units).

## Cross-references

- [Preconditioners](../preconditioners/index.md)
- [Parameterization](../parameterization/index.md)
