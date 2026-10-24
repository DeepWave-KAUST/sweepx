# Truncated Gauss-Newton

`sweepx.opt.optim.TruncatedGaussNewton` is the truncated Gauss-Newton
optimizer for non-linear least-squares problems
$L(\theta) = \tfrac{1}{2}\|r(\theta)\|^2$ - the standard data-misfit form
of FWI.

It approximates the Hessian by the Gauss-Newton matrix $J^\top J$
(dropping the second-derivative-of-residual term) and solves
$(J^\top J + \mu I)\, d = -g$ inexactly via inner PCG, where $g = J^\top r$.

The matrix $J^\top J$ is symmetric positive semidefinite, so inner CG never
hits negative curvature - much more well-behaved than the full Newton
matrix on indefinite problems. Optional Levenberg-Marquardt damping $\mu$
adds robustness when $J^\top J$ is rank-deficient.

## Calling convention - **closure returns the residual**

`TruncatedGaussNewton.step()` accepts a `residual_fn` that returns the
residual tensor $r(\theta)$, **not** the loss. The optimizer forms
$L = \tfrac{1}{2}\|r\|^2$ and computes both $g$ and the GN matrix-vector
product internally via JVP / VJP. As with `TruncatedNewton`, **do not
call `backward` inside `residual_fn`**.

The JVP is a double backward through the residual, so the solver must be
twice differentiable. With sweep that is `impl='eager'` with `torch.compile`
off; the setup's default solver (`impl='c'` on a GPU) refuses a second
backward over the same graph with a `RuntimeError`:

```python
from sweep.propagator.options import EagerOptions, Full

# A twice-differentiable solver: the eager backend, all time steps kept in
# memory, no torch.compile.
solver = PropTorch(Acoustic(device=dev), shape=(96, 128), dh=10.0, dt=1e-3, device=dev,
                   impl="eager", memory=Full(), eager_options=EagerOptions(use_compile=False))
```

The eager solver is much slower than `impl='c'`: on a V100 this 5-epoch
example took about 14 minutes, against under a second for the L-BFGS example
on `impl='c'`:

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.TruncatedGaussNewton([vp], bounds=(1500.0, 5500.0), max_inner_iter=5)

def residual_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])
    return (syn - obs).reshape(-1)            # 1-D residual

for _ in range(cfg["epochs"]):
    opt.step(residual_fn)
```

As for Truncated Newton, the inner CG solves for the step, so the line search
starts at $\alpha = 1$ in parameter units; trials are projected onto
`bounds` before they are evaluated.

## API

```python
sweepx.opt.optim.TruncatedGaussNewton(
    params,
    *,
    bounds=None,
    preconditioner=None,
    damping=0.0,                     # Levenberg-Marquardt mu; 0.0 => pure GN
    forcing="eisenstat_walker",      # "eisenstat_walker" | "constant"
    forcing_constant=1e-2,
    max_inner_iter=50,
    c1=1e-4, c2=0.9,
    alpha_init=1.0, alpha_max=10.0,
    max_ls_iter=25,
)
```

Diagnostics:

- `opt.last_inner_iters` - inner PCG iterations in the most recent step.
- `opt.last_forcing` - tolerance used in the most recent step.

## When to use

TGN is the right tool when the misfit is genuinely a sum of squares and the
Gauss-Newton approximation is close to the true Hessian (most of FWI fits
this). On problems where the residual is large and the second-order term
matters, TruncatedNewton may converge faster despite its more delicate
inner solve.

For very ill-conditioned problems, set ``damping > 0`` (Levenberg-Marquardt
style) to stabilize the inner CG.

## References

- Metivier, L., Brossier, R., Virieux, J., and Operto, S. (2013). "Full
  waveform inversion and the truncated Newton method." _SIAM Journal on
  Scientific Computing_ 35(2), B401-B437.
  [DOI:10.1137/120877854](https://doi.org/10.1137/120877854).
- Levenberg, K. (1944). "A method for the solution of certain non-linear
  problems in least squares." _Quarterly of Applied Mathematics_ 2(2),
  164-168.
  [DOI:10.1090/qam/10666](https://doi.org/10.1090/qam/10666).
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of
  nonlinear parameters." _Journal of the Society for Industrial and Applied
  Mathematics_ 11(2), 431-441.
  [DOI:10.1137/0111030](https://doi.org/10.1137/0111030).
