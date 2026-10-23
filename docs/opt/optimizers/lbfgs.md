# L-BFGS-FWI

`sweepx.opt.optim.LBFGS` is the limited-memory BFGS quasi-Newton optimizer
(Nocedal 1980; Nocedal & Wright 2006, Algorithm 7.4) with three FWI-friendly
extensions on top of `torch.optim.LBFGS`:

1. **Bounds.** ``bounds=(lo, hi)`` enforces a box on every parameter: each
   line-search trial point is projected onto it before it is evaluated.
   (The active-set Byrd-Lu-Nocedal L-BFGS-B variant is
   [`LBFGSB`](lbfgsb.md); the two share the same FWI-friendly knobs and
   differ only in how bounds are handled.)
2. **Preconditioner as $H_0$.** A preconditioner $P$ can be supplied as the
   initial inverse-Hessian inside the two-loop recursion. With the
   [`PseudoHessianPrecond`](../preconditioners/pseudo_hessian.md), this gives
   *preconditioned L-BFGS* - faster convergence on FWI's poorly-scaled
   Hessian without giving up any of the L-BFGS history.
3. **Restart.** ``restart_every=N`` clears the stored ``(s, y)`` history
   every ``N`` iterations. Useful for stochastic / source-encoded FWI
   loops where the curvature pairs become stale faster than the iterate.

## Two-loop recursion

At iteration $k$, with stored pairs $\{(s_i, y_i, \rho_i)\}_{i=k-m}^{k-1}$:

```
q := g_k
for i = k-1, ..., k-m:
    alpha_i := rho_i * s_i^T q
    q       := q - alpha_i * y_i
r := H_0 q
for i = k-m, ..., k-1:
    beta := rho_i * y_i^T r
    r    := r + (alpha_i - beta) * s_i
return d_k := -r
```

where $H_0 = \gamma_k I$ with the standard Liu-Nocedal scalar
$\gamma_k = (s_{k-1}^\top y_{k-1}) / (y_{k-1}^\top y_{k-1})$, or, with a
preconditioner, $H_0 = \gamma_k P$ with
$\gamma_k = (s_{k-1}^\top y_{k-1}) / (y_{k-1}^\top P y_{k-1})$, so the step
keeps the scale the curvature pairs measured whatever the scale of $P$. The
step length is then chosen by strong-Wolfe, starting at $\alpha = 1$.

Without stored pairs (the first step, after a restart) the direction is
$-g$ or $-Pg$, whose size follows the gradient, not the parameters. For a
velocity model in m/s that is far too small, so pass `initial_step=` (in
parameter units): the first trial then changes the largest component by
exactly that much, and the search may expand it up to
`alpha_max / alpha_init` (10) times. Without it, a search that ends at
`alpha_max` or leaves the parameters unchanged emits a `RuntimeWarning`.

## API

```python
sweepx.opt.optim.LBFGS(
    params,                          # iterable of torch.Tensor
    *,
    history_size=10,                 # m in N&W
    bounds=None,                     # (lo, hi) projected box
    preconditioner=None,             # callable used as H_0
    line_search="strong_wolfe",
    c1=1e-4, c2=0.9,                 # Wolfe constants (quasi-Newton default)
    alpha_init=1.0, alpha_max=10.0,  # multiply d = -H_k g
    max_ls_iter=25,
    initial_step=None,               # first trial without history, parameter units
    restart_every=None,
    strict_descent=True,             # fall back to -g if curvature lies
)
```

A closure is **required**. Diagnostic: `opt.history_length` returns the
current count of stored ``(s, y)`` pairs.

## Drop-in usage

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
precond = sweepx.opt.precond.PseudoHessianPrecond(eps=1e-5)
solver.compute_illumination = True   # impl='c': backward() fills the illumination maps

opt = sweepx.opt.optim.LBFGS(
    [vp],
    history_size=10,
    bounds=(1500.0, 5500.0),
    preconditioner=precond,
    initial_step=20.0,               # first trial step: 20 m/s
    restart_every=20,
)

def closure():
    opt.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    # Refresh the preconditioner from the illumination of this backward.
    S, R = solver.source_illumination, solver.receiver_illumination
    precond.update([vp], source_illum=S / S.max(), receiver_illum=R / R.max())
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

`step()` calls the closure at the current model first, so the direction is
always preconditioned with the illumination of that model. The illumination
maps exist only on sweep's native backend (`impl='c'`); see the
[pseudo-Hessian page](../preconditioners/pseudo_hessian.md) for why they are
normalized and how `eps` is chosen.

## When to use

L-BFGS is the de-facto standard for FWI: it usually converges much faster
than NCG and Steepest Descent, and a pseudo-Hessian preconditioner can speed
it up further.

For ill-posed problems where the Hessian has near-zero eigenvalues, consider
Truncated Newton - it can solve the indefinite Newton system more
robustly via inner CG with explicit forcing.

## References

- Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
  _Mathematics of Computation_ 35(151), 773-782.
  [DOI:10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7).
- Liu, D. C., and Nocedal, J. (1989). "On the limited memory BFGS method
  for large scale optimization." _Mathematical Programming_ 45, 503-528.
  [DOI:10.1007/BF01589116](https://doi.org/10.1007/BF01589116).
- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer, Section 7.2.
  [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
