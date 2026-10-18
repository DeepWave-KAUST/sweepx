# Optimizers

This page lists every optimizer shipped by `sweepx.opt`, grouped by category.

| Algorithm | Class | Category | Closure required? | Bounds | Preconditioner |
|---|---|---|---|---|---|
| [Steepest Descent](steepest_descent.md) | `SteepestDescent` | First-order | only with `line_search` | yes | yes |
| [Nonlinear CG (PR+)](ncg.md) | `NCG` | First-order | yes | yes | yes |
| [L-BFGS-FWI](lbfgs.md) | `LBFGS` | Quasi-Newton | yes | yes (projected) | yes (as `H_0`) |
| [Truncated Newton (HF)](newton.md) | `TruncatedNewton` | Newton-type | yes (loss-only) | yes | yes (PCG inner) |
| [Truncated Gauss-Newton](gauss_newton.md) | `TruncatedGaussNewton` | Newton-type | yes (residual) | yes | yes (PCG inner) |
| [Trust-Region (TTR/ASTR)](trust_region.md) | `TrustRegion` | Trust-region | yes (loss-only) | yes | yes (PCG inner, TTR) |
| [L-BFGS-B](lbfgsb.md) | `LBFGSB` | Quasi-Newton | yes | yes (active-set) | yes (as `H_0`) |
| [Adam-FWI](adam_fwi.md) | `AdamFWI` | Adaptive | optional | yes | n/a (per-param adaptive) |
| [Lion](lion.md) | `Lion` | Adaptive (sign-momentum) | optional | yes | n/a |
| [SAM](sam.md) | `SAM` | Wrapper (sharpness-aware) | yes | inherited | inherited |
| [Shampoo](shampoo.md) | `Shampoo` | Second-order (Kronecker) | optional | yes | n/a (Kronecker-factored built in) |
| [SOAP](soap.md) | `SOAP` | Second-order (Shampoo + Adam) | optional | yes | n/a |

See each algorithm's own page for the API, theory, and references.

## Conventions

Every optimizer in `sweepx.opt`:

1. Subclasses `torch.optim.Optimizer`, so it works with any code that already
   uses `torch.optim.Adam`/`SGD`/`LBFGS`.
2. Implements `step(closure=None)` with the same closure semantics as
   `torch.optim.LBFGS`: closure must `zero_grad()`, recompute the loss, call
   `backward()`, and return the loss.
3. Accepts an optional `bounds=(lo, hi)` argument; if provided, parameters are
   projected to the box after each update.
4. Accepts an optional `preconditioner=` argument that pre-multiplies gradients
   before the step. See [Preconditioners](../preconditioners/index.md).

## Cross-references

- [Preconditioners](../preconditioners/index.md)
- [Parameterization](../parameterization/index.md)
