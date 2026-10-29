# Neural reparameterizations (sweepx.nn)

Implicit FWI: invert the weights of a coordinate network that renders the
model, instead of the grid. Python: `sweepx.nn.VelocityINR`. Task YAML: a
`reparam:` block on an `fwi` task.

## Contents

- [VelocityINR in Python](#velocityinr-in-python) (runnable)
- [The reparam: block](#the-reparam-block) (runnable)
- [Settings that matter](#settings-that-matter)
- [Other modules](#other-modules)

## VelocityINR in Python

`vp = base + vp_std * net(coords) + vp_mean` (or `vp_std * net + vp_mean`
with `direct_velocity=True`). `base` is a non-trainable buffer (the starting
model); the net's output is O(1), so **`vp_std` (m/s) sets how far the model
can move** and acts as the effective step size together with the lr.

```python
import torch
from sweepx.nn import VelocityINR

dev = "cuda" if torch.cuda.is_available() else "cpu"
base = torch.full((40, 80), 2000.0, device=dev)                       # (nz, nx), m/s
net = VelocityINR(base, vp_std=500.0,
                  hidden_features=64, hidden_layers=3,
                  use_hash_encoding=True, hash_levels=8,
                  hash_base_resolution=4, hash_finest_resolution=64).to(dev)
vp = net()                                                            # (40, 80), starts near base
print(vp.shape, float((vp - base).abs().max().detach()))

target = base.clone(); target[15:25, 30:50] = 2400.0                  # toy: fit a model directly
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
for it in range(30):
    loss = (net() - target).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"{loss.item():.1f}", sum(p.numel() for p in net.parameters()))
```

In an FWI loop replace the toy loss by the solver misfit:
`pred = solver(wavelet, sources, receivers, models=[net()])`.

## The reparam: block

```python
import numpy as np
from scipy.ndimage import gaussian_filter
vp = np.full((40, 80), 2000.0, np.float32); vp[20:, :] = 2400.0; vp[12:22, 30:50] = 2300.0
np.save("vp_true.npy", vp); np.save("vp_init.npy", gaussian_filter(vp, 6).astype(np.float32))
```

```yaml
# file: ifwi.yaml
task_type: fwi
task_id: demo_ifwi
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 400}
wavelet: {kind: ricker, fm: 10.0, delay: 0.1}
geometry:
  kind: line
  sources:   {step: 20, depth: 2, start: 10}
  receivers: {step: 2,  depth: 2}
physics: {equation: Acoustic}
backend: {impl: eager, eager_options: {use_compile: false}}
init_model: {name: vp, path: vp_init.npy}
obs: {synthetic_from: {name: vp, path: vp_true.npy}}
reparam:
  kind: velocity_inr
  lr: 1.0e-4               # the network's lr (default 1e-4)
  vp_std: 500.0            # perturbation scale in m/s: size it to the update the data needs
  direct_velocity: false   # learn a perturbation on top of init_model
  hidden_features: 64
  hidden_layers: 3
  use_bias: false          # hash features in [0, 1): bias hurts here
  hash: {enabled: true, levels: 8, base_resolution: 4, finest_resolution: 64}
  save_net: true           # write reparam_net.pt (+ optimizer state) for a warm start
optimizer: {kind: adam, lr: 25.0}   # required; only kind/eps/betas are used under reparam
epochs: 2
batchsize: 4
```

```bash
sweepx tasks run ifwi.yaml
ls runs/demo_ifwi runs/demo_ifwi/output
```

## Settings that matter

| key | guidance |
|---|---|
| `reparam.lr` | network lr, default 1e-4 (stage scale: `lr_scale` × `inr_lr_scale`). `optimizer.lr` is unused |
| `vp_std` | the effective step size in m/s. Too small saturates: on the Marmousi demo (true update rms 357 m/s, range ±1400) `vp_std` 300 gave RMSE 325 vs 267 at 2500 (sweepx docs, example 07) |
| `hash.levels`, `base_resolution`, `finest_resolution` | finest resolution ≈ the finest feature (cells) you want; per-axis lists `[n_z, n_x]` / `[n_z, n_y, n_x]` for anisotropic grids |
| `hash.c2f` | coarse-to-fine level unfreezing (`base_levels`, `ramp`, `warmup`, `ramp_end`, `final_levels` per stage); `growing: true` allocates fine levels lazily (not with `backend: triton`) |
| `fourier` | alternative global encoding (exclusive with `hash`) |
| `coord_min`, `coord_max` | keep 0 / 1 with the hash grid (it expects coordinates in [0, 1)); nothing checks this |
| `backward_mode` | `two_pass_full` (default), `two_pass_chunked` (+ `backward_chunk_rows`, less memory), `single_step` (one graph; OOM-prone, and refused with illumination/smoothing on plan streaming) |
| `free_params` | invert several models from one trunk: list `{name, std, bounds, water_value}` in the equation's model order, vp first |
| `init_from`, `init_optimizer_from: auto` | warm start from a previous run's `reparam_net.pt` (+ its optimizer state; a missing file is an error). Hash config must match |
| `lateral_downsample`, `compile_render` | speed knobs for large 3-D renders |
| `water:` (top level) | the network renders `water.velocity` in the water; the reparam water keys are retired |

`qc.gradient_png` is skipped under reparam (the gradient lives on the weights).

## Other modules

| class | use |
|---|---|
| `SIREN`, `SirenMLP`, `SineLayer` | sine-activation MLPs (`first_omega0`, `hidden_omega0`, default 30) |
| `MultiResHashGrid` (`have_triton()` for the fused kernel), `GrowingHashGrid` | Instant-NGP hash encoders, 2-D/3-D, coords in [0, 1) |
| `MultiParamINR` | one network, several models (vp, vs, rho, ...) |
| `TVPrior(order, x/y/z_weight)` | smoothness penalty; YAML `smooth_regularization:` |
| `DIPReparam`, `sweepx.opt.parameterization.DIPParameterization` | deep image prior |
| `sweepx.nn.diffusion` (`UNet2D`, `GaussianDiffusion`, `EMA`, `DiffusionVelocityPrior`) | DDPM prior; YAML `diffusion_prior: {ckpt_path, weight, kind: red, ...}` needs a trained checkpoint |
| `SirenWavelet` | 1-D SIREN representation of a source wavelet |
