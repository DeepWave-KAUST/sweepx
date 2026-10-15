# Installation

```bash
pip install sweepx
```

`sweepx.tasks` is part of `sweepx`, together with the modules it builds on:
`sweepx.loss` (misfits), `sweepx.io` (SEG-Y, acquisition plans) and `sweepx.nn`
(network reparameterisations and priors). The install also pulls the solver
(`sweep-solver`) and puts the `sweep-tasks` command on your PATH (also reachable
as `sweepx tasks`).

Field data in SEG-Y needs the `segy` extra:

```bash
pip install "sweepx[segy]"
```

Verify the install:

```python
import sweepx.tasks
print(sweepx.tasks.__version__)
```

`sweepx.tasks` runs anywhere `sweep` runs — a CPU for small jobs, CUDA for
production. The GPU backend (`backend: {impl: c}`) ships precompiled in the
`sweep-solver` wheel, so nothing compiles at install or on first use;
`impl: eager` is pure PyTorch. [Example 15](../examples/introspect.md) lists
which equations your build has compiled.
