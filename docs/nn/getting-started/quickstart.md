# Quickstart

## A network instead of a grid

`VelocityINR` renders a velocity model from a network. It starts from a background
model and learns the perturbation on top of it:

```python
import torch
from sweep_nn import VelocityINR

vp_init = torch.full((141, 341), 3000.0, device="cuda")       # (nz, nx), m/s
net = VelocityINR(vp_init, vp_std=400.0,                       # perturbation scale, m/s
                  hidden_features=128, hidden_layers=6,
                  use_hash_encoding=False).cuda()              # a plain SIREN
vp = net()                                                     # (141, 341) tensor, m/s
```

`vp_std` sets how far the network can move the model: the network output is of order
one, so the update is of order `vp_std` m/s. Size it to the perturbation the data needs.
`use_hash_encoding=True` puts a multiresolution hash grid in front of the SIREN, which
fits fine detail with a much shallower network.

## In an FWI loop

Optimize the network weights, not the grid. Any differentiable solver works; with sweep:

```python
optim = torch.optim.Adam(net.parameters(), lr=1e-4)
for it in range(n_iters):
    vp = net()
    pred = solver(wavelet, sources, receivers, models=[vp])
    loss = (pred - observed).pow(2).mean()
    optim.zero_grad(); loss.backward(); optim.step()
```

The [examples](../examples/README.md) run this loop end to end on Overthrust and
Marmousi, against conventional FWI.

## Without a wave equation

[`quickstart.py`](https://github.com/DeepWave-KAUST/sweep-nn/blob/main/docs/examples/quickstart.py)
fits a SIREN straight to a velocity model, the shortest way to see the API. It runs on a
CPU in a few seconds.
