# Adam-FWI

`sweepx.opt.optim.AdamFWI` is a single class wrapping four Adam-family
update rules with FWI-friendly knobs. Choose the rule via the `variant`
flag:

| `variant` | Rule | Reference |
|---|---|---|
| `"adam"`  | Vanilla Adam | Kingma & Ba 2014 |
| `"adamw"` | Adam with **decoupled** weight decay | Loshchilov & Hutter 2017 |
| `"radam"` | Rectified Adam (variance-controlled lr in early steps) | Liu et al. 2019 |
| `"nadam"` | Adam with Nesterov momentum | Dozat 2016 |

On top of the stock formulas, every variant accepts:

- `bounds=(lo, hi)` projected after each step,
- `mask` tensor (or callable returning one) that scales every update,
  weight decay included, so zeros pin components (water-layer freeze, ROI
  mask, ...),
- `amsgrad=True` to use the running max of the second moment,
- default `eps=1e-22` matching sweep's existing FWI Adam baseline.

## API

```python
sweepx.opt.optim.AdamFWI(
    params,
    lr=1e-3,
    *,
    variant="adam",                 # "adam" | "adamw" | "radam" | "nadam"
    betas=(0.9, 0.999),
    eps=1e-22,
    weight_decay=0.0,
    amsgrad=False,
    bounds=None,
    mask=None,                      # Tensor, dict {id(p): Tensor}, or callable p -> Tensor
)
```

A closure is **optional** (the closure path matches `torch.optim.Adam`'s
behaviour: it just recomputes the loss and returns it for tracking).

As for `torch.optim.Adam`, the update is normalized by the running gradient
magnitude, so `lr` is roughly the change per step in parameter units (m/s for
a velocity model), whatever the scale of the gradient.

## Drop-in usage

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)

# Water-layer freeze: the top 10 grid rows (depth is axis 0 of (nz, nx)) do not update.
mask = torch.ones_like(vp); mask[:10, :] = 0.0

opt = sweepx.opt.optim.AdamFWI(
    [vp],
    lr=cfg["lr"],
    variant="adam",
    bounds=(1500.0, 5500.0),
    mask=mask,
)

for _ in range(cfg["epochs"]):
    opt.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    opt.step()
```

## Variant choice

- **`adam`** is the safe default and matches the baseline sweep uses today.
- **`adamw`** is preferred when you also want regularization toward zero
  (or toward a background, via `PerturbationFromBackground`). Its decay is
  `lr * weight_decay` of the parameter per step, so on a velocity in m/s it
  pulls the whole model toward 0 m/s: `lr=10`, `weight_decay=1e-4` take
  2 m/s per step off a 2000 m/s cell. Use it on a perturbation instead.
- **`radam`** stabilizes the early steps when the gradient variance is
  high (e.g. stochastic source encoding with very small batches).
- **`nadam`** trades a little extra cost for slightly faster convergence
  in some experiments; differences from `adam` are usually small on FWI.

## References

- Kingma, D. P., and Ba, J. (2015). "Adam: A method for stochastic
  optimization."
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Loshchilov, I., and Hutter, F. (2019). "Decoupled weight decay
  regularization." _ICLR 2019_.
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J.
  (2019). "On the variance of the adaptive learning rate and beyond."
  [arXiv:1908.03265](https://arxiv.org/abs/1908.03265).
- Dozat, T. (2016). "Incorporating Nesterov momentum into Adam."
  [openreview/OM0jvwB8jIp57ZJjtNEZ](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ).
- Reddi, S. J., Kale, S., and Kumar, S. (2018). "On the convergence of
  Adam and beyond."
  [arXiv:1904.09237](https://arxiv.org/abs/1904.09237) (AMSGrad).
