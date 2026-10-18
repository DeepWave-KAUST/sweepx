# Installation

```bash
pip install sweep-tasks
```

That pulls the solver (`sweep-solver`) and the three packages sweep-tasks builds
on: `sweep-loss` (misfits), `sweep-io` (SEG-Y, acquisition plans) and
`sweep-nn` (network reparameterisations and priors). Or install the whole
ecosystem with the umbrella package:

```bash
pip install sweepx
```

Verify the install:

```python
import sweep_tasks
print(sweep_tasks.__version__)
```

`sweep-tasks` runs anywhere `sweep` runs — a CPU for small jobs, CUDA for
production. The GPU backend (`backend: {impl: c}`) ships precompiled in the
`sweep-solver` wheel, so nothing compiles at install or on first use;
`impl: eager` is pure PyTorch. [Example 15](../examples/introspect.md) lists
which equations your build has compiled.
