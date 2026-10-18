# SOAP - Adam in Shampoo's eigenbasis

`sweep_opt.optim.SOAP` (Vyas et al. 2024) runs Adam in the eigenbasis of
Shampoo's preconditioner. It follows the
[official implementation](https://github.com/nikhilvyas/SOAP) (commit
`a1e5535`) step for step; `tests/test_soap_parity.py` checks parameters and
optimizer state against that code bit for bit.

Each parameter keeps one Shampoo factor per tensor axis ($L, R$ for a
matrix), a moving average of the gradient contracted over the other axes,

$$
L \leftarrow \beta_s L + (1-\beta_s)\, G G^\top,\qquad
R \leftarrow \beta_s R + (1-\beta_s)\, G^\top G,
$$

and Adam runs on the gradient rotated into their eigenbases $Q_L, Q_R$:

$$
\tilde G = Q_L^\top G Q_R,\qquad
\text{Adam moments } m, v \text{ of } \tilde G,\qquad
W \leftarrow W - \alpha\, Q_L \frac{\hat m}{\sqrt{\hat v} + \epsilon} Q_R^\top .
$$

The factors are gradient outer products (Adagrad's preconditioner), not a
Hessian; no Hessian-vector products are computed.

Order of operations (as in the official code):

1. **First step**: the factors are built from the gradient and
   eigendecomposed; the parameter is **not** updated, so a gradient never
   enters its own projection.
2. **Later steps**: Adam step in the current basis, then decoupled weight
   decay, then the factors are updated with the gradient. Every
   `update_freq` steps the basis is refreshed by one power iteration + QR
   from the previous basis (Algorithm 4 of the paper), after sorting its
   columns by Rayleigh quotient; $v$ is permuted with the columns and not
   rotated, $m$ is carried into the new basis exactly.

An axis longer than `max_precond_dim` gets no factor (identity), so a
`(1e5, 1)` hash-grid table is updated by Adam. 1-D parameters are plain Adam
unless `precondition_1d=True`; `merge_dims=True` merges neighbouring axes
until their product would exceed `max_precond_dim`.

## API

```python
sweep_opt.optim.SOAP(
    params,
    lr=3e-3,
    *,
    betas=(0.95, 0.95),
    shampoo_beta=-1.0,               # factors' EMA; < 0 uses betas[1]
    eps=1e-8,
    weight_decay=0.0,                # decoupled (AdamW); official default 0.01
    update_freq=10,                  # official name: precondition_frequency
    max_precond_dim=10000,           # longer axes get no factor
    merge_dims=False,
    precondition_1d=False,
    normalize_grads=False,
    correct_bias=True,
    bounds=None,
)
```

All hyper-parameters except `bounds` are per parameter group, so param
groups and `torch.optim.lr_scheduler` work as usual.

Differences from the official code: `weight_decay` defaults to 0 (FWI
parameters such as a velocity grid must not be decayed); the factors are kept
in the parameter dtype, so float64 parameters work (the official code fails
on them), and only float32/float64 are accepted; `closure` runs with
gradients enabled; `bounds` is sweep-opt's projection after the step; there
is no `data_format` option.

## When to use

The paper evaluates SOAP only on language-model pre-training, where it beats
AdamW and Shampoo, with the largest gains at large batch sizes. The defaults
above are the official ones, tuned for that setting.

These defaults need not transfer to FWI-type problems with small batches;
tune `lr`, `betas` and `update_freq` per problem (`betas=(0.9, 0.999)` with
`update_freq=1` is a reasonable start).

The rotation suits the weight matrices of a network. For parameters laid out
on a spatial grid (a velocity model, a hash-encoding table) it equalises the
step over spatial wavenumbers and amplifies weakly constrained short
wavelengths; keep such parameters unrotated (an axis longer than
`max_precond_dim` is left unrotated, i.e. plain Adam) or use Adam for them.

## References

- Vyas, N., Morwani, D., Zhao, R., Kwun, M., Shapira, I., Brandfonbrener,
  D., Janson, L., and Kakade, S. (2024). "SOAP: Improving and stabilizing
  Shampoo using Adam."
  [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).
  Official implementation: <https://github.com/nikhilvyas/SOAP>.
- Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
  stochastic tensor optimization."
  [PMLR](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf).
