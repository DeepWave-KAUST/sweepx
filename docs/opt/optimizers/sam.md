# SAM - Sharpness-Aware Minimization

`sweep_opt.optim.SAM` wraps **any base optimizer** in Foret et al.'s (2021)
two-step ascent-then-descent update:

$$
\epsilon_w = \rho \, \frac{g}{\|g\|},\qquad
\theta_{\text{tmp}} = \theta + \epsilon_w,\qquad
g_{\text{tmp}} = \nabla f(\theta_{\text{tmp}}),
$$

then call the base optimizer's `step()` using $g_{\text{tmp}}$ at the
*original* $\theta$. The result is a step that descends toward
**flatter** minima - empirically more robust on a wide range of problems.

## API

```python
sweep_opt.optim.SAM(
    base_optimizer,                 # an Optimizer instance (torch or sweep-opt)
    *,
    rho=0.05,                       # ascent radius
    eps=1e-12,                      # gradient-normalization stabilizer
)
```

`SAM.step()` requires a closure and **calls it twice** per step (once for
$g$, once for $g_{\text{tmp}}$).

## Drop-in usage

```python
import sweep_opt

vp = init_vp.clone().requires_grad_(True)
base = sweep_opt.optim.LBFGS([vp], history_size=10, bounds=(1500.0, 5500.0))
opt = sweep_opt.optim.SAM(base, rho=0.05)

def closure():
    base.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

## When to use

SAM costs ~2x per step (two forward / backward passes) and helps when the
loss surface has many local minima of varying sharpness. For FWI it is a
research-mode optimizer; the wall-clock penalty is significant on real
problems.

## References

- Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. (2021).
  "Sharpness-aware minimization for efficiently improving generalization."
  _ICLR 2021_.
  [arXiv:2010.01412](https://arxiv.org/abs/2010.01412).
