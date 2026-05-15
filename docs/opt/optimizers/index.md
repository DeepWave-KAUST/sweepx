# Optimizers

This page lists every optimizer shipped by `sweep-opt`, grouped by category
and tagged by the Tier in which it lands.

| Algorithm | Class | Category | Tier | Closure required? | Bounds | Preconditioner |
|---|---|---|---|---|---|---|
| [Steepest Descent](steepest_descent.md) | `SteepestDescent` | First-order | 1 | only with `line_search` | yes | yes |
| [Nonlinear CG (PR+)](ncg.md) | `NCG` | First-order | 1 | yes | yes | yes |
| [L-BFGS-FWI](lbfgs.md) | `LBFGS` | Quasi-Newton | 1 | yes | yes (projected) | yes (as `H_0`) |
| [Truncated Newton (HF)](newton.md) | `TruncatedNewton` | Newton-type | 2 | yes (loss-only) | yes | yes (PCG inner) |
| [Truncated Gauss-Newton](gauss_newton.md) | `TruncatedGaussNewton` | Newton-type | 2 | yes (residual) | yes | yes (PCG inner) |

The matrix is appended to as new optimizers land. See each algorithm's own
page for the API, theory, and references.

## Conventions

Every optimizer in `sweep-opt`:

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
- [Marmousi report](../report.md) - convergence comparisons on the same problem
