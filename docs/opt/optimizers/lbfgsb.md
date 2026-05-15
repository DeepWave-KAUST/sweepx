# L-BFGS-B

`sweep_opt.optim.LBFGSB` is the **bound-constrained** limited-memory BFGS
algorithm of Byrd, Lu, and Nocedal (1995). Unlike the projected-gradient
variant in :doc:`L-BFGS-FWI <lbfgs>`, it identifies an *active set* at
each iteration and updates the BFGS history only on the *free* subspace
- preventing curvature pairs from mixing components that are pinned to
the box.

This implementation uses gradient-projection with active-set masking
rather than the full Cauchy-point + subspace minimization of the
canonical Fortran implementation. On the smooth bounded quadratic
benchmarks it matches `scipy.optimize.minimize(method="L-BFGS-B")` to
line-search tolerance, which is enough for FWI's per-grid-point velocity
bounds.

## Active-set rule

A component $i$ is in the active set at iteration $k$ iff:

- $\theta_i = \theta_i^{\text{lo}}$ and $g_i > 0$, **or**
- $\theta_i = \theta_i^{\text{hi}}$ and $g_i < 0$.

Search-direction components in the active set are zeroed; the line
search is projected onto the box at every trial.

## API

```python
sweep_opt.optim.LBFGSB(
    params,
    *,
    bounds=(lo, hi),                # required
    history_size=10,
    preconditioner=None,            # optional H_0
    line_search="strong_wolfe",
    c1=1e-4, c2=0.9,
    alpha_init=1.0, alpha_max=10.0,
    max_ls_iter=25,
    restart_every=None,
    active_tol=0.0,
)
```

A closure is required.

## When to use

Use `LBFGSB` when **the optimum lies on (or near) the bound** for a
non-trivial number of components. The projected-gradient `LBFGS` variant is
fine when the bounds are "soft" (only a few components ever bind), but
becomes inefficient when many components are active because the BFGS
history poisoned by zeroed projected updates produces poor search
directions.

For FWI: a water-layer freeze (top rows clamped to `1500.0`) is a typical
case where `LBFGSB` outperforms `LBFGS`.

## References

- Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). "A limited memory
  algorithm for bound constrained optimization." _SIAM Journal on
  Scientific Computing_ 16(5), 1190-1208.
  [DOI:10.1137/0916069](https://doi.org/10.1137/0916069).
- Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). "Algorithm 778:
  L-BFGS-B: Fortran subroutines for large-scale bound-constrained
  optimization." _ACM Trans. Math. Soft._ 23(4), 550-560.
  [DOI:10.1145/279232.279236](https://doi.org/10.1145/279232.279236).
