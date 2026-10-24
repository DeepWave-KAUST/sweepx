# Data convention

`sweepx.opt` is backend-agnostic - it operates on whatever `torch.Tensor`
parameters you pass to its optimizers, and never touches your data tensors.

That said, the library is designed to compose cleanly with `sweep` and
`sweepx.loss`, which share a common shot-data layout:

```
(nshots, nt, nreceivers, nchannel)
```

- `nshots` - number of independent source experiments in the mini-batch
- `nt` - number of time samples (axis = -3, the **time axis**)
- `nreceivers` - number of receivers per shot
- `nchannel` - number of recorded components (1 for pressure, 2/3 for elastic, ...)

## Parameter tensors

Inversion parameters in `sweep`-style FWI are model tensors with grid shape:

- 2-D: `(nz, nx)` for `vp`, `vs`, `rho`, ... (depth is axis 0)
- 3-D: `(nz, ny, nx)` ditto.

`sweepx.opt` optimizers do not enforce a specific shape; they call
`param.flatten()` internally where needed. Bound projection,
preconditioning, and re-parameterization all preserve the original tensor
shape so the user code never has to reshape.

## Illumination tensors

Pseudo-Hessian-style preconditioners need source/receiver illumination maps
shaped like the model grid. `sweep` computes them only on its native backend
(`impl='c'`) and only when `solver.compute_illumination = True` (off by
default): each `backward()` then fills `solver.source_illumination` and
`solver.receiver_illumination`. They are `None` after a forward without
gradient and zeros between a forward and its `backward()`; the eager backend
never produces them. sweepx.opt's `PseudoHessianPrecond.update(...)` accepts
them as keyword arguments; see
[hooking it into a sweep run](../preconditioners/pseudo_hessian.md#hooking-it-into-a-sweep-run).
