# Installation

```bash
pip install sweep-tasks    # also pulls sweep (the solver) + pydantic + torch + pyyaml
```

Or via the umbrella meta-package, which bundles the whole ecosystem:

```bash
pip install sweepx
```

Verify the install:

```python
import sweep_tasks
print(sweep_tasks.__version__)
```

`sweep-tasks` runs anywhere `sweep` runs — CPU for small jobs, CUDA for
production. The GPU backend (`backend: {kind: c}`) is JIT-compiled by `sweep`
on first use and needs `nvcc >= 12.4`; the eager backend needs no compiler.
