# Installation

```bash
pip install sweep-nn
```

That needs only PyTorch and NumPy. It also comes with `pip install sweepx`, through
sweep-tasks.

The [examples](../examples/README.md) drive sweep's compiled wave solver, so they also
need a CUDA GPU and

```bash
pip install sweep-solver
```

Verify the install:

```python
import sweepx.nn
print(sweepx.nn.__version__)
```

The hash grid has an optional fused Triton kernel
(`MultiResHashGrid(..., backend="triton")`); `sweepx.nn.have_triton()` says whether it
is available. The default `backend="pytorch"` runs anywhere PyTorch does.
