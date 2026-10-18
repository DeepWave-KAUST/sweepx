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

```python
def residual_fn():
    syn = solver(wavelet, sources, receivers, models=[vp])
    return (syn - obs).reshape(-1)            # 1-D residual

opt.step(residual_fn)
```

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
