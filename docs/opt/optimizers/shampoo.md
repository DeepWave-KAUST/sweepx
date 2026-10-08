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

The inverse roots are refreshed every `update_freq` steps (default 1).

With $k$ factored axes the exponent is $-1/(2k)$: a vector (or a
`(1, n)` / `(n, 1)` matrix) uses $L^{-1/2}$, i.e. full-matrix AdaGrad.
An axis longer than `max_precond_dim` gets no factor; a parameter with no
factored axis (e.g. a `(1e5, 1)` hash-grid table) falls back to diagonal
AdaGrad, so no $d \times d$ matrix is ever allocated for it.
Higher-order tensors are flattened to 2-D along the leading axis.

## API

```python
sweep_opt.optim.Shampoo(
    params,
    lr=1e-2,
    *,
    eps=1e-4,                        # ridge, relative to each factor's largest eigenvalue
    update_freq=1,                   # eigendecomp every N steps
    max_precond_dim=4096,            # longer axes get no factor
    bounds=None,
)
```

All hyper-parameters except `bounds` are per parameter group, so param
groups and `torch.optim.lr_scheduler` work as usual. Because `eps` is
relative, the update does not depend on the overall gradient scale (FWI
gradients often have $\|GG^\top\| \sim 10^{-10}$, which an absolute ridge
would swamp). The eigendecomposition runs in float64 with one jittered
retry; if it still fails the previous root is kept.

## Caveats

- Shampoo's preconditioner is the **sum** of outer products of past
  gradients; the implicit step size therefore decays like $1/\sqrt{t}$.
  This is different from Adam (which uses an exponential moving average)
  and means Shampoo loss curves often plateau later than Adam's at the
  same `lr`.
- With `update_freq > 1` the cached root can amplify a gradient
  direction that appeared after the last refresh by up to
  $\epsilon^{-1/(2k)}$; with a small `eps` this diverges. Use
  `eps=1e-2` or larger when refreshing lazily.
- For large parameters where $d_1$ or $d_2$ is in the thousands, the
  $O(d^3)$ eigendecomposition cost dominates; reach for SOAP or a
  blocked variant.

## References

- Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
  stochastic tensor optimization." _ICML 2018_.
  [PMLR](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf).
