# Bounds parameterizations

`sweep_opt.parameterization` ships three smooth reparameterizations that
let you optimize over a latent variable while the FWI solver still sees
the physical model:

- :class:`SigmoidBounds` - $v(z) = v_{\min} + (v_{\max} - v_{\min})\,\sigma(z)$
- :class:`LogVelocity` - $v(z) = e^{z}$ (requires $v > 0$)
- :class:`PerturbationFromBackground` - $v(z) = v_0 + z$ around a smooth
  background $v_0$

Each is a :class:`Parameterization` with a round-tripped `forward` /
`inverse` pair, so you can switch the optimizer's variable from the
physical model to the latent without rewriting the FWI loop.

## API contract

```python
class Parameterization:
    def forward(self, latent: Tensor) -> Tensor:
        """Map latent -> physical."""

    def inverse(self, physical: Tensor) -> Tensor:
        """Map physical -> latent (so latent_0 = inverse(initial_model))."""
```

The library guarantees `inverse(forward(z)) == z` and (modulo clamps)
`forward(inverse(p)) == p`.

## Drop-in usage

```python
import sweep_opt
from sweep_opt.parameterization import SigmoidBounds

# Initialise the latent so that forward(latent) == initial_model.
param = SigmoidBounds(vmin=1500.0, vmax=5500.0)
latent = param.inverse(initial_model).detach().requires_grad_(True)

opt = sweep_opt.optim.LBFGS([latent], history_size=10)

def closure():
    opt.zero_grad()
    physical = param.forward(latent)
    syn = solver(wavelet, sources, receivers, models=[physical])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

This pattern composes naturally with any sweep-opt optimizer; the
optimizer iterates the unbounded `latent` while the data misfit sees the
bounded physical model. Gradients flow through `forward` automatically via
PyTorch autograd.

## When to use which

- **SigmoidBounds**: bounds are a hard physical constraint (water velocity
  $\geq 1500$, max sediment $\leq 5500$) and you do not want to deal with
  active-set bookkeeping. Pairs well with :class:`LBFGS` (without
  `bounds=`), :class:`NCG`, or :class:`Adam-FWI`.
- **LogVelocity**: stability and scale-invariance for strictly positive
  models. The optimizer sees `log(v)` so a 1% perturbation has the same
  step size at v=1500 as at v=5500.
- **PerturbationFromBackground**: most natural for "low-resolution
  background + high-resolution update" workflows; the latent is small
  and the optimizer iterates a residual.

## References

- Tarantola, A. (1987). _Inverse Problem Theory_. Elsevier.
  ISBN 9780444427656. (Discusses log / exp parameterizations for
  positive-definite quantities.)
- Lempitsky, V., Vedaldi, A., and Ulyanov, D. (2018). "Deep Image Prior."
  _CVPR 2018_.
  [DOI:10.1109/CVPR.2018.00984](https://doi.org/10.1109/CVPR.2018.00984)
  (the most general reparameterization in the library, see
  :doc:`dip`).
