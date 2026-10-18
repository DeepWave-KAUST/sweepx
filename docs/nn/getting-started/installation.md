# Installation

```bash
pip install sweepx
```

`sweepx.nn` is part of `sweepx`; on its own it needs only PyTorch and NumPy. The
[examples](../examples/README.md) drive sweep's compiled wave solver (installed with
`sweepx`), so they also need a CUDA GPU.

Verify the install:

```python
import sweepx.nn
print(sweepx.nn.__version__)
```

The hash grid has an optional fused Triton kernel
(`MultiResHashGrid(..., backend="triton")`); `sweepx.nn.have_triton()` says whether it
is available. The default `backend="pytorch"` runs anywhere PyTorch does.
