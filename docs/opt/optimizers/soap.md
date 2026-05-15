# SOAP - Shampoo eigenbasis + Adam updates

`sweep_opt.optim.SOAP` (Vyas et al. 2024) maintains Shampoo-style
preconditioners $L, R$ for a 2-D parameter, diagonalises them every
`update_freq` steps to obtain rotations $Q_L, Q_R$, and runs **Adam updates
inside the rotated eigenbasis**:

$$
\tilde G = Q_L^\top G Q_R,\qquad
\text{(Adam state $m, v$ in the rotated basis)},\qquad
M_{t+1} = M_t - \alpha\, Q_L\, \tilde u\, Q_R^\top
$$

where $\tilde u = m/\hat b_1 \big/ \sqrt{v / \hat b_2 + \epsilon}$ is the
standard Adam update.

Unlike Shampoo, SOAP uses an exponential moving average for both $L, R$
and the Adam moments; its implicit step size does not decay with $t$.

## API

```python
sweep_opt.optim.SOAP(
    params,
    lr=1e-3,
    *,
    betas=(0.9, 0.999),
    eps=1e-8,
    update_freq=50,                  # eigendecomp every N steps
    bounds=None,
)
```

## When to use

SOAP is the most modern of the second-order ML optimizers in sweep-opt
and is competitive with Adam on most benchmarks while keeping similar
memory footprint per step (one extra rotation matrix per axis is small
unless dimensions get large). For FWI it is research-grade; expect to
tune `lr` and `update_freq` per problem.

## References

- Vyas, N., Morwani, D., Zhao, R., Kaplun, G., Kakade, S., and
  Shanker, S. (2024). "SOAP: Improving and stabilizing Shampoo using
  Adam."
  [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).
- Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
  stochastic tensor optimization."
  [PMLR](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf).
