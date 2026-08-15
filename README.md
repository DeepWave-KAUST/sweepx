# sweepx

**The `sweep` seismic wave-equation ecosystem — one install, then `import sweep`.**

[![PyPI](https://img.shields.io/pypi/v/sweepx.svg)](https://pypi.org/project/sweepx/)
[![Python](https://img.shields.io/pypi/pyversions/sweepx.svg)](https://pypi.org/project/sweepx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`sweepx` is the umbrella install for [`sweep`](https://github.com/DeepWave-KAUST/sweep) —
a GPU seismic wave-equation engine (forward modelling, RTM, FWI/LSRTM) — and its
companion packages. It carries **no code of its own**: `pip install sweepx` pulls
in the engine and the published companions so you can `import sweep` and go.

```bash
pip install sweepx
```

## The ecosystem

Every piece is a **separate, independently-installable package** (the same layout
as the [PyLops](https://github.com/PyLops) family). `sweepx` bundles the published
ones; you can also `pip install` any single package on its own.

### Published — pulled in by `sweepx`

| Package | `import` | What it does |
|---|---|---|
| [`sweep-solver`](https://github.com/DeepWave-KAUST/sweep) | `sweep` | Wave-equation **engine**: equations (acoustic / elastic / VTI / TTI / VRZ / SEM), propagators (torch / JAX / CUDA `impl='c'`), operators, boundary-saving, and FWI / LSRTM / RTM building blocks. |
| [`sweep-agent`](https://github.com/DeepWave-KAUST/sweep-agent) | `sweep_agent` | Natural-language **control layer** — chat + files → `sweep`, through a local LLM (Ollama / vLLM). Runs forward modelling (acoustic **and** elastic), loads benchmark models, plots — all on the base install. `pip install "sweep-agent[ui]"` adds a web UI. |

### Companions — not on PyPI yet

Shipping as they mature; when a companion is published it is added to `sweepx`'s
dependencies, so you keep the same `pip install sweepx`.

| Package | What it does |
|---|---|
| `sweep-tasks` | Production **FWI / LSRTM runner** — spec schemas, YAML configs, losses, optimizers, multi-GPU, IO. |
| `sweep-loss` | Misfit / loss functions. |
| `sweep-nn` | Neural reparameterizations (INR / hash / SIREN encoders). |
| `sweep-tomo` | First-arrival **traveltime tomography** (eikonal + SIRT / FATT). |
| *(planned)* | `sweep-io`, `sweep-viz`, `sweep-preproc`, `sweep-opt`. |

## Quick start

Drive the engine directly:

```python
import sweep
from sweep.propagator.torch import PropTorch     # PyTorch propagator
from sweep import equations, propagator          # native submodules
```

…or in plain language, via the agent (needs a local LLM):

```bash
sweep-agent chat
>>> load the Marmousi benchmark model and run a forward — show the shot gather
```

## Import name

After `pip install sweepx`, **import as `sweep`**, not `sweepx` — same pattern as
`pip install scikit-learn` → `import sklearn`. The import name `sweep` was already
taken on PyPI, so the installable is named `sweepx`.

## CUDA backend

`sweep`'s GPU backend (`impl='c'`) is **JIT-compiled against your own PyTorch on
first use** — so a single wheel works with **any** torch version and any Python 3,
with no prebuilt CUDA/torch/Python matrix. It needs a CUDA GPU and `nvcc >= 12.4`
(a system toolkit, `module load cuda`, or `conda install -c nvidia cuda-toolkit`):

```python
import sweep
sweep.precompile()   # optional: build the CUDA backend now (~3-5 min, then cached)
```

Drop `precompile()` and the compile happens automatically on first use of
`impl='c'`, cached thereafter in `~/.cache/torch_extensions`. The pure-Python
eager / JAX backends need no nvcc.

```python
import sweep
print(sweep.is_torch_binding_available())          # torch + CUDA GPU + nvcc present?
print(sweep.backend.torch.binding.diagnostics())   # usable / reason / cuda_home / built
```

## License

MIT — see [LICENSE](LICENSE).
