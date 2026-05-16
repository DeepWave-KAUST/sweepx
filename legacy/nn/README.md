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
| `sweep_nn.siren` | SIREN-style implicit neural representation |
| `sweep_nn.dip` | Deep Image Prior — small U-Net fed by a fixed latent |
| `sweep_nn.priors` | Learned prior wrappers (feature extractor for perceptual losses) |

## Install

```bash
pip install sweep-nn
```

Or via the ecosystem meta-package: `pip install sweep[full]`.

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
