# SAM - Sharpness-Aware Minimization

`sweepx.opt.optim.SAM` wraps a **closure-free base optimizer** in Foret et
al.'s (2021) two-step ascent-then-descent update:

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
sweepx.opt.optim.SAM(
    base_optimizer,                 # an Optimizer whose step() needs no closure
    *,
    rho=0.05,                       # ascent radius, parameter units (L2 norm)
    eps=1e-12,                      # gradient-normalization stabilizer
)
```

`SAM.step()` requires a closure and **calls it twice** per step (once for
$g$, once for $g_{\text{tmp}}$). It then calls `base_optimizer.step()`
**without** a closure, so the base must update from the gradient in `.grad`:
Adam-FWI, Lion, Shampoo, SOAP, Steepest Descent without a line search, or
`torch.optim.SGD` / `Adam`. A base that needs a closure (L-BFGS, L-BFGS-B,
NCG, Steepest Descent with a line search, the Newton-type and trust-region
optimizers, `torch.optim.LBFGS`) raises `TypeError` at construction.

`rho` is the Euclidean norm of the perturbation $\epsilon_w$ over all
parameters, in parameter units, so it grows with the model size: on the
setup model, `rho=50.0` moves the largest cell by about 3 m/s.

## Drop-in usage

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
base = sweepx.opt.optim.AdamFWI([vp], lr=cfg["lr"], bounds=(1500.0, 5500.0))
opt = sweepx.opt.optim.SAM(base, rho=50.0)

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
