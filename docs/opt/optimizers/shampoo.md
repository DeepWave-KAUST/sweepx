# Shampoo

`sweep_opt.optim.Shampoo` is the **Kronecker-factored 4th-root**
preconditioner of Gupta et al. (2018). For a 2-D parameter
$M \in \mathbb{R}^{d_1 \times d_2}$ it maintains two running
preconditioners

$$
L_t = L_{t-1} + G_t G_t^\top \quad (d_1 \times d_1),\qquad
R_t = R_{t-1} + G_t^\top G_t \quad (d_2 \times d_2),
$$

and applies the update

$$
M_{t+1} = M_t - \alpha\, L_t^{-1/4}\, G_t\, R_t^{-1/4}.
$$

The 4th-root inversions are cached and refreshed every `update_freq`
steps (default 50) to amortize the eigendecomposition cost.

For 1-D parameters Shampoo reduces to a single $L^{-1/4}$ along the only
axis. Higher-order tensors are flattened to 2-D along the leading axis.

## API

```python
sweep_opt.optim.Shampoo(
    params,
    lr=1e-2,
    *,
    eps=1e-6,                        # ridge added inside the matrix root
    update_freq=50,                  # eigendecomp every N steps
    bounds=None,
)
```

## Caveats

- Shampoo's preconditioner is the **sum** of outer products of past
  gradients; the implicit step size therefore decays like $1/\sqrt{t}$.
  This is different from Adam (which uses an exponential moving average)
  and means Shampoo loss curves often plateau later than Adam's at the
  same `lr`.
- For large parameters where $d_1$ or $d_2$ is in the thousands, the
  $O(d^3)$ eigendecomposition cost dominates; reach for SOAP or a
  blocked variant.

## References

- Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
  stochastic tensor optimization." _ICML 2018_.
  [PMLR](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf).
