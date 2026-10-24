# Trust-Region (TTR / ASTR)

`sweepx.opt.optim.TrustRegion` is the trust-region Newton-type optimizer.
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
takes a `loss_fn` returning the loss tensor **without calling backward**.

The Hessian-vector products are double backward passes through the
solver, so the solver must be twice differentiable. With sweep that is
`impl='eager'` with `torch.compile` off; the setup's default solver
(`impl='c'` on a GPU) returns a gradient without an autograd graph, and the
compiled eager step does not support a double backward either:

```python
from sweep.propagator.options import EagerOptions, Full

# A twice-differentiable solver: the eager backend, all time steps kept in
# memory, no torch.compile.
solver = PropTorch(Acoustic(device=dev), shape=(96, 128), dh=10.0, dt=1e-3, device=dev,
                   impl="eager", memory=Full(), eager_options=EagerOptions(use_compile=False))
```

With `impl='c'`, `step()` raises a `RuntimeError` that names the solver. It
cannot catch every case: with a nonlinear parameterization such as
[`SigmoidBounds`](../parameterization/bounds.md), the gradient keeps a graph
through the parameterization, the Hessian-vector product runs, and it is
wrong (on the setup model it pointed nearly orthogonally to the eager one).
The eager solver is much slower than `impl='c'`: on a V100 the 5-epoch
examples of the Truncated Newton, Truncated Gauss-Newton and Trust-Region
pages took 12-14 minutes each, against under a second for the L-BFGS
example on `impl='c'`.

The trust radius is the Euclidean norm of the step over all parameters, in
parameter units: a 10 m/s change on each of $10^4$ cells has norm $10^3$.
With the defaults (`initial_radius=1.0`, `max_radius=100.0`) the first step
of a model in m/s has a norm of at most 1 m/s over the whole grid, and no
step ever exceeds 100, so set both:

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.TrustRegion(
    [vp], variant="ttr", bounds=(1500.0, 5500.0),
    initial_radius=300.0, max_radius=1e4, max_inner_iter=5,
)

def loss_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])
    return (syn - obs).pow(2).sum()

for _ in range(cfg["epochs"]):
    opt.step(loss_fn)
```

A trial step whose loss is not finite (e.g. a solver driven past its
stability limit) is rejected and shrinks the radius; with `bounds`, the trial
point is projected onto the box before it is evaluated.

## API

```python
sweepx.opt.optim.TrustRegion(
    params,
    *,
    variant="ttr",                  # "ttr" | "astr"
    bounds=None,
    preconditioner=None,            # used only by TTR (inside Steihaug-CG)
    initial_radius=1.0,             # L2 norm of the step, parameter units
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
