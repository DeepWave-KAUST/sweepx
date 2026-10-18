# Quick start

`sweep-opt` exposes every optimizer through PyTorch's standard
`torch.optim.Optimizer` interface, so the entire library can be adopted by
swapping a single line.

## The drop-in replacement

Any PyTorch FWI loop already looks like this:

```python
import torch
optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

for epoch in range(cfg["epochs"]):
    optimizer.zero_grad()
    syn  = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    optimizer.step()
```

To use any first-order sweep-opt optimizer, change only the constructor
(everything Tier-1+ supports a closure-less `step()` for compatibility with
this exact loop):

```python
import sweep_opt
optimizer = sweep_opt.optim.SteepestDescent([inv_vp], lr=cfg["lr"])
```

For algorithms that need a closure (line-search variants, L-BFGS, Newton-type,
trust-region) wrap the loss block in a function and pass it to `step()`:

```python
def closure():
    optimizer.zero_grad()
    syn  = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    return loss

for epoch in range(cfg["epochs"]):
    optimizer.step(closure)
```

## With bounds and a preconditioner

```python
import sweep_opt

precond = sweep_opt.precond.PseudoHessianPrecond(eps=1e-3)
optimizer = sweep_opt.optim.LBFGS(
    [inv_vp],
    history_size=10,
    bounds=(1500.0, 5500.0),         # m/s velocity bounds
    preconditioner=precond,
    line_search="strong_wolfe",
    restart_every=20,
)
```

The available algorithms grow tier-by-tier; see the
[optimizer overview](../optimizers/index.md).
