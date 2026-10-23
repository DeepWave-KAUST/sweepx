# Truncated Newton (Hessian-Free Newton)

`sweepx.opt.optim.TruncatedNewton` solves the Newton system

$$
H_k\, d_k = -g_k
$$

approximately, by inner preconditioned conjugate-gradient (PCG) iterations
that use **matrix-free Hessian-vector products**. It then chooses the step
length $\alpha_k$ by strong-Wolfe.

The "truncation" is governed by a **forcing sequence** $\{\eta_k\}$ that
sets the relative tolerance on the inner CG residual:

- ``forcing="eisenstat_walker"`` (default): adaptive
  $\eta_k = \min(0.5, \sqrt{\|g_k\|})$ shrinks as the gradient shrinks
  (Eisenstat & Walker 1996).
- ``forcing="constant"``: a single user-supplied tolerance.

When the Hessian has near-zero or negative eigenvalues (FWI's ill-posed
modes), inner CG either truncates early or, in the negative-curvature case,
falls back to the steepest-descent direction. This makes TN more robust
than direct Newton on ill-posed inverse problems.

## Calling convention - **closure must NOT call `backward`**

Unlike the rest of sweepx.opt's optimizers, `TruncatedNewton` requires the
intact autograd graph to assemble HVPs, so the user-supplied callable is a
pure **loss function** (no `backward`):

```python
def loss_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])   # depends on vp (requires_grad)
    return (syn - obs).pow(2).sum()                          # do NOT call .backward()
```

The optimizer internally calls `torch.autograd.grad(loss, params, create_graph=True)`
to build the gradient with the graph alive, then differentiates the
$g \cdot v$ inner product to get $H v$.

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

## API

```python
sweepx.opt.optim.TruncatedNewton(
    params,
    *,
    bounds=None,
    preconditioner=None,
    forcing="eisenstat_walker",     # "eisenstat_walker" | "constant"
    forcing_constant=1e-2,          # used when forcing="constant"
    max_inner_iter=50,
    c1=1e-4, c2=0.9,
    alpha_init=1.0, alpha_max=10.0,
    max_ls_iter=25,
)
```

Diagnostics:

- `opt.last_inner_iters` - PCG iterations used in the most recent step.
- `opt.last_forcing` - tolerance $\eta_k$ used in the most recent step.

## Drop-in usage

With the eager `solver` above:

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.TruncatedNewton(
    [vp],
    bounds=(1500.0, 5500.0),
    forcing="eisenstat_walker",
    max_inner_iter=5,
)

def loss_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])
    return (syn - obs).pow(2).sum()

for _ in range(cfg["epochs"]):
    opt.step(loss_fn)
```

The inner CG solves for the step itself, so the line search starts at
$\alpha = 1$ in parameter units and needs no `initial_step`; trials are
projected onto `bounds` before they are evaluated. `preconditioner=` takes
any `sweepx.opt` preconditioner for the inner PCG, but the
[pseudo-Hessian](../preconditioners/pseudo_hessian.md) needs illumination
maps, which sweep fills only on `impl='c'`: the eager solver does not produce
them.

## When to use

TN is the most robust choice for **ill-posed, indefinite, or
strongly-nonlinear** FWI problems where L-BFGS struggles to build a useful
quasi-Newton approximation. The cost per outer step is
$O(k_{\text{inner}})$ matrix-vector products, so for cheap forward solvers
TN can win; for expensive ones (e.g. 3-D acoustic FWI with deep models),
L-BFGS often wins on wall-clock.

## References

- Nash, S. G. (2000). "A survey of truncated-Newton methods."
  _Journal of Computational and Applied Mathematics_ 124(1-2), 45-59.
  [DOI:10.1016/S0377-0427(00)00426-X](https://doi.org/10.1016/S0377-0427(00)00426-X).
- Eisenstat, S. C., and Walker, H. F. (1996). "Choosing the forcing terms
  in an inexact Newton method." _SIAM Journal on Scientific Computing_
  17(1), 16-32.
  [DOI:10.1137/0917003](https://doi.org/10.1137/0917003).
- Pearlmutter, B. A. (1994). "Fast exact multiplication by the Hessian."
  _Neural Computation_ 6(1), 147-160.
  [DOI:10.1162/neco.1994.6.1.147](https://doi.org/10.1162/neco.1994.6.1.147).
- Hestenes, M. R., and Stiefel, E. (1952). "Methods of conjugate gradients
  for solving linear systems." _Journal of Research of NBS_ 49(6), 409-436.
  [DOI:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044).
