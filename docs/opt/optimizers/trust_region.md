# Trust-Region (TTR / ASTR)

`sweep_opt.optim.TrustRegion` is the trust-region Newton-type optimizer.
Two subproblem solvers are exposed via the `variant` flag:

- ``"ttr"`` (default): **truncated CG inside the trust radius** (Steihaug
  1983). Works well in high dimensions; matrix-free; preconditionable.
- ``"astr"`` : **adaptive subspace** trust region (Wang & Rao 2015). The
  subproblem is solved on the 2-D subspace
  $\mathrm{span}(g_k, d_{k-1})$ via a 2x2 dense solve. Cheap and
  surprisingly competitive on FWI-shaped problems.

Both variants follow the standard trust-region acceptance recipe:

$$
\rho = \frac{f(\theta_k) - f(\theta_k + d)}{m_k(0) - m_k(d)},
\qquad
m_k(d) = f_k + g_k^\top d + \tfrac{1}{2} d^\top H_k d.
$$

If $\rho > \eta_{\text{acc}}$ the step is accepted; otherwise it is rejected
and the trust radius shrinks. A "very good" step ($\rho > 0.75$) at the
trust boundary doubles the radius (capped at ``max_radius``).

## Calling convention

Like `TruncatedNewton` and `TruncatedGaussNewton`, `TrustRegion.step()`
takes a `loss_fn` returning the loss tensor **without calling backward**:

```python
def loss_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])
    return (syn - obs).pow(2).sum()

opt.step(loss_fn)
```

## API

```python
sweep_opt.optim.TrustRegion(
    params,
    *,
    variant="ttr",                  # "ttr" | "astr"
    bounds=None,
    preconditioner=None,            # used only by TTR (inside Steihaug-CG)
    initial_radius=1.0,
    max_radius=100.0,
    eta_acceptable=0.1,             # accept if rho > this
    eta_increase=0.75,              # grow radius if rho > this and at boundary
    radius_shrink=0.25,             # delta *= this on shrink
    radius_grow=2.0,                # delta *= this on grow
    max_inner_iter=50,              # Steihaug-CG cap (TTR)
    cg_tol=1e-2,                    # Steihaug-CG residual tolerance (TTR)
)
```

Diagnostics:

- `opt.trust_radius` - current $\Delta_k$.
- `opt.last_rho` - $\rho$ from the most recent step.
- `opt.last_accepted` - whether the last step was accepted.
- `opt.last_boundary` - whether the last step landed on the trust boundary.

## When (not) to use

Trust-region methods shine when the objective has regions of poor
quadratic-model fit (high non-linearity, indefinite Hessians, sharp
features). The radius adaptation lets the optimizer probe the model's
trustworthiness rather than relying on a line-search safeguard.

For pure FWI L2 loss with reasonably smooth velocity models, L-BFGS or
Truncated Gauss-Newton usually win on wall-clock; trust-region helps when
the data are noisy, the misfit is non-convex (envelope, AWI), or you need
implicit regularization on the step size.

## References

- Steihaug, T. (1983). "The conjugate gradient method and trust regions in
  large scale optimization." _SIAM J. Numer. Anal._ 20(3), 626-637.
  [DOI:10.1137/0720042](https://doi.org/10.1137/0720042).
- Wang, Y., and Rao, Y. (2015). "Reflection seismic waveform tomography in
  transversely isotropic media." _Geophysical Journal International_ 202(2),
  1362-1373.
  [DOI:10.1093/gji/ggv226](https://doi.org/10.1093/gji/ggv226).
- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer, Algorithms 4.1 and 7.2.
  [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
