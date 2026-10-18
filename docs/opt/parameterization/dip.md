# DIP-FWI

`sweep_opt.parameterization.DIPParameterization` represents the physical
model as the output of a (typically randomly-initialized) neural-network
generator. The optimizer iterates the **latent** input to the generator
(and, optionally, the generator's weights), while the forward solver
consumes ``generator(latent)``. The implicit regularization comes from
the *spectral bias* of the architecture: low-frequency structure is
fitted faster than high-frequency noise (Lempitsky et al. 2018).

For FWI specifically, He et al. (2024) showed that a U-Net generator
gives smooth velocity updates without an explicit Tikhonov / TV penalty.

## API

```python
sweep_opt.parameterization.DIPParameterization(
    generator,                       # any nn.Module
    latent_shape,                    # the latent's tensor shape
    *,
    out_min=None,                    # if both supplied, output is sigmoid-bounded
    out_max=None,
)
```

A reference small generator is shipped:

```python
sweep_opt.parameterization.SmallUNet2D(
    in_channels=16,
    out_channels=1,
    base_channels=16,
)
```

It produces an output of the same spatial shape as the input - choose a
latent whose spatial dims match the desired model grid (or interpolate
after).

## Drop-in usage

```python
import torch, sweep_opt

generator = sweep_opt.parameterization.SmallUNet2D(
    in_channels=8, out_channels=1, base_channels=32,
)
param = sweep_opt.parameterization.DIPParameterization(
    generator,
    latent_shape=(1, 8, 64, 128),    # batch + channels + (nx, nz)
    out_min=1500.0, out_max=5500.0,  # bounded vp
)
latent = param.random_latent(scale=0.1)
opt = sweep_opt.optim.AdamFWI(
    [latent, *generator.parameters()],
    lr=1e-3,
    variant="adamw", weight_decay=1e-4,
)

for _ in range(cfg["epochs"]):
    opt.zero_grad()
    vp = param.forward(latent).squeeze(0).squeeze(0)
    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    opt.step()
```

## Caveats

- ``inverse`` is **not** generally invertible for an arbitrary generator;
  initialise ``latent`` randomly and accept that ``generator(latent_init)``
  does not match the initial physical model. If you need a particular
  starting model, fit the generator to it offline first.
- The choice of generator architecture matters more than for any other
  parameterization in the library; see He et al. 2024 for FWI-specific
  recommendations.

## References

- Lempitsky, V., Vedaldi, A., and Ulyanov, D. (2018). "Deep Image Prior."
  _CVPR 2018_.
  [DOI:10.1109/CVPR.2018.00984](https://doi.org/10.1109/CVPR.2018.00984).
- He, T., et al. (2024). "Deep image prior full-waveform inversion."
  _JGR: Machine Learning and Computation_ 1, e2024JH000227.
  [DOI:10.1029/2024JH000227](https://doi.org/10.1029/2024JH000227).
