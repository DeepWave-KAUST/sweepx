# Lion

`sweepx.opt.optim.Lion` is the **sign-momentum** optimizer of Chen et al.
(2023). It uses the sign of a momentum-blended gradient as the update
direction:

$$
c_t = \mathrm{sign}\bigl(\beta_1 m_{t-1} + (1 - \beta_1) g_t\bigr),
\qquad
\theta_t = \theta_{t-1} - \alpha\,\bigl(c_t + \lambda \theta_{t-1}\bigr),
\qquad
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t.
$$

Compared to Adam, Lion holds **half the per-parameter state** (only `m`,
no `v`) and matches or exceeds Adam on a wide range of vision and
language benchmarks. Less is known about its behaviour on physics-based
inverse problems; sweepx.opt ships it for FWI experiments.

## API

```python
sweepx.opt.optim.Lion(
    params,
    lr=1e-4,                # use a 3-10x smaller lr than Adam
    *,
    betas=(0.9, 0.99),
    weight_decay=0.0,
    bounds=None,
)
```

A closure is **optional**.

## Drop-in usage

The update is a sign, so `lr` is the change of every component per step, in
parameter units (m/s here):

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.Lion([vp], lr=2.0, bounds=(1500.0, 5500.0))

for _ in range(cfg["epochs"]):
    opt.zero_grad()
    loss = (solver(wavelet, sources, receivers, models=[vp]) - obs).pow(2).sum()
    loss.backward()
    opt.step()
```

## Notes

- Lion's typical `lr` is ~3-10x smaller than Adam's because the sign
  step is unit-scale per component.
- Memory: one momentum buffer per parameter, vs Adam's two. Substantial
  savings for large 3-D FWI problems.
- No bias correction (Chen et al.'s ablations showed it hurts).

## References

- Chen, X., et al. (2023). "Symbolic discovery of optimization algorithms."
  [arXiv:2302.06675](https://arxiv.org/abs/2302.06675).
