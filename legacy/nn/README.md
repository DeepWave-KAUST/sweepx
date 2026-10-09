# sweep-nn

PyTorch neural reparameterization and priors for seismic FWI. **PyTorch-only.**
No sweep / no jax dependency.

## The core idea

In neural FWI you don't update the velocity grid `vp` directly. You define
a small network `Net` and a latent `z`, and update *those* via the loss:

```
vp = Net(z)                      # implicit / learned reparameterization
pred = forward_solver(vp, ...)
loss = misfit(pred, obs) + lambda * regularizer(Net.parameters())
loss.backward()
optimizer.step()                 # updates Net's weights and / or z
```

This package provides reusable `Net` candidates and helpers around them.

## What's in it

| Module | What it gives you |
|---|---|
| `sweep_nn.reparam` | `Reparameterizer` base class — anything mapping ``() -> vp_tensor`` |
| `sweep_nn.siren` | SIREN-style implicit neural representation ([Sitzmann 2020](https://doi.org/10.48550/arXiv.2006.09661)); `SIREN` (coord-based, squashes to ``[vp_min, vp_max]``), `SirenMLP` (generic ``in→out`` building block), `SineLayer` |
| `sweep_nn.hash_encoding` | Multi-resolution hash-grid encoder ([Müller et al. 2022](https://doi.org/10.1145/3528223.3530127)) — `MultiResHashGrid`, pure PyTorch, 2-D and 3-D |
| `sweep_nn.velocity_inr` | `VelocityINR` — hash → SIREN → ``base + delta`` velocity field. The recommended FWI reparameterization. Supports ``update_base_velocity()`` for multi-stage transitions without resetting learned params |
| `sweep_nn.wavelet` | `SirenWavelet` — 1-D SIREN for learning a source wavelet from time |
| `sweep_nn.dip` | Deep Image Prior — small U-Net fed by a fixed latent |
| `sweep_nn.priors` | Learned prior wrappers (feature extractor for perceptual losses) |

## Hash-encoded SIREN for FWI

`VelocityINR` is the production-grade reparam choice for FWI:

```python
import torch
from sweep_nn import VelocityINR

init_vp = torch.from_numpy(np.load("init_vp.npy"))   # (nz, nx)
net = VelocityINR(
    init_vp,
    vp_std=50.0,                # typical delta magnitude in m/s
    bounds=(1450.0, 5500.0),    # render-time clamp
    hash_levels=16, hash_finest_resolution=512,
    hidden_features=64, hidden_layers=3,
)
optim = torch.optim.Adam(net.parameters(), lr=1e-4)

for it in range(n_iters):
    vp = net()                  # forward through hash + SIREN
    pred = solver(wavelet, sources, receivers, models=[vp])
    loss = misfit(pred, obs)
    optim.zero_grad(); loss.backward(); optim.step()

# Multi-scale FWI: swap the base when the grid resolution changes —
# the network's learned parameters carry over.
net.update_base_velocity(init_vp_finer)
```

The hash encoder (`Instant-NGP` style) gives the SIREN head a structured
multi-resolution coordinate basis, so a small MLP can fit high-frequency
velocity detail with O(table_size) parameters instead of O(grid_points).
A typical config (`L=16, F=2, T=2^15`) is ~1M parameters total, regardless
of the velocity grid size.

## Install

```bash
pip install sweep-nn
```

It also comes with `pip install sweepx`, through sweep-tasks.

## Quick example

```python
import torch
from sweep_nn.siren import SIREN

shape = (200, 400)                # (nz, nx)
net = SIREN(out_shape=shape, vp_min=1500.0, vp_max=4500.0)
optim = torch.optim.Adam(net.parameters(), lr=1e-4)

for it in range(n_iters):
    vp = net()                    # (nz, nx) tensor in m/s
    pred = solver(wavelet, sources, receivers, models=[vp])
    loss = misfit(pred, obs)
    optim.zero_grad(); loss.backward(); optim.step()
```

## Design notes

- **Output is always in physical units.** Networks internally produce
  normalized features; the `Reparameterizer` base scales them to
  ``[vp_min, vp_max]`` before returning. No re-scaling boilerplate on
  the caller side.
- **No global state.** Each `Reparameterizer` instance owns its latent
  (or none, if it generates from coordinates directly). Serialize with
  the usual `state_dict()` / `load_state_dict()`.

## License

MIT.
