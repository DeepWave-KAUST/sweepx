# sweep-nn

Neural reparameterizations and priors for full-waveform inversion, in plain PyTorch.
Instead of updating the velocity grid directly, implicit FWI updates the weights of a
network that renders the grid:

```
vp  = net()                       # e.g. vp_init + vp_std · SIREN(hash(z, x))
pred = solver(wavelet, sources, receivers, models=[vp])
loss = misfit(pred, obs)
loss.backward()                   # the gradient flows through vp into the weights
```

The network's own smoothness regularizes the inversion, and the model has far fewer
free parameters than grid cells. `sweep-nn` has no dependency on the wave solver: any
differentiable PyTorch solver works, and the examples use
[sweep](../solver/index.md).

Installed with `pip install sweep-nn` (also bundled by `pip install sweepx`)
→ `import sweep_nn`.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __[Getting started](getting-started/installation.md)__

    ---

    Install, then fit a SIREN to a velocity model in a few lines.

-   :material-notebook-outline: __[Examples](examples/README.md)__

    ---

    Implicit FWI on Overthrust and Marmousi, reproducing the *Geophysics* paper.

-   :material-api: __[API reference](api/index.md)__

    ---

    `VelocityINR`, the hash grid, SIREN, the priors.

</div>

## What is in it

| Module | What it gives you |
|---|---|
| `sweep_nn.velocity_inr` | `VelocityINR`: a hash-encoded SIREN rendering `vp_init + vp_std · net(coords)`, 2-D or 3-D. The recommended reparameterization for FWI. |
| `sweep_nn.hash_encoding` | `MultiResHashGrid`, the Instant-NGP multiresolution hash grid ([Müller et al. 2022](https://doi.org/10.1145/3528223.3530127)), pure PyTorch with an optional Triton kernel |
| `sweep_nn.siren` | `SIREN`, `SirenMLP`, `SineLayer` ([Sitzmann et al. 2020](https://doi.org/10.48550/arXiv.2006.09661)) |
| `sweep_nn.multi_param_inr` | `MultiParamINR`: one network for several models at once (vp, vs, rho, …) |
| `sweep_nn.priors` | `TVPrior`, `SeabedFreezeMask`, learned-prior wrappers |
| `sweep_nn.diffusion` | A DDPM/DDIM velocity prior: `UNet2D`/`UNet3D`, `GaussianDiffusion`, and `DiffusionVelocityPrior`, which turns a trained checkpoint into a plug-and-play (RED) regularizer |
| `sweep_nn.wavelet` | `SirenWavelet`: a 1-D SIREN for a source wavelet |
| `sweep_nn.dip` | `DIPReparam`: a deep image prior, a small U-Net fed by a fixed latent |
