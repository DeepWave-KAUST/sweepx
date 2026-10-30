# sweepx

**The `sweep` toolkit: seismic I/O, FWI losses, neural reparameterizations, the YAML
task runner and a natural-language agent, on top of the `sweep` wave-equation engine.**

[![PyPI](https://img.shields.io/pypi/v/sweepx.svg)](https://pypi.org/project/sweepx/)
[![Python](https://img.shields.io/pypi/pyversions/sweepx.svg)](https://pypi.org/project/sweepx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-sweepx.deepwave.group-blue)](https://sweepx.deepwave.group/)

```bash
pip install sweepx
```

That installs the toolkit and the engine it runs on,
[`sweep-solver`](https://github.com/DeepWave-KAUST/sweep) (`import sweep`), which ships
prebuilt CUDA cores. Optional extras: `sweepx[segy]`, `[hdf5]`, `[animate]`,
`[agent-ui]`, `[mcp]`, `[vllm]`, `[all]`.
[Installation guide](https://sweepx.deepwave.group/getting-started/).

## What is in it

| Module | What it does | Docs |
|---|---|---|
| `sweepx.io` | Seismic I/O — SEG-Y, acquisition geometry, data and model plans. | [IO](https://sweepx.deepwave.group/io/) |
| `sweepx.loss` | Misfit functions for FWI — two dozen, from L2 to optimal transport. | [Loss](https://sweepx.deepwave.group/loss/) |
| `sweepx.nn` | Neural reparameterizations and priors — INR / hash / SIREN encoders, TV and diffusion priors. | [NN](https://sweepx.deepwave.group/nn/) |
| `sweepx.opt` | FWI optimizers — L-BFGS, truncated Newton / Gauss-Newton, Adam, Shampoo / SOAP, preconditioners and bound parameterizations. | [Opt](https://sweepx.deepwave.group/opt/) |
| `sweepx.tasks` | Production **FWI / LSRTM / RTM runner** — typed YAML task specs, a CLI, multi-GPU runs, field-data pipelines, preprocessing and plotting. | [Tasks](https://sweepx.deepwave.group/tasks/) |
| `sweepx.agent` | Natural-language **control layer** — chat + files → validated runs, through a local LLM (Ollama / vLLM). | [Agent](https://sweepx.deepwave.group/agent/) |

The modules are layered: `agent` → `tasks` → `io` / `loss` / `nn` / `opt`. A module
imports only from the layers below it, and the four libraries neither import each other
nor the engine (checked in CI with import-linter).

## Quick start

```bash
sweep-tasks run examples/tasks/synthetic/09_forward_constant_box.yaml   # or: sweepx tasks run …
sweep-tasks init fwi -o my_fwi.yaml    # an annotated FWI template (any task type) ...
sweep-tasks run my_fwi.yaml            # ... that runs as written: Marmousi, nothing to download
```

```python
import sweep                         # the engine: equations, propagators
from sweepx.loss import EnvelopeLoss # the toolkit, one module at a time
from sweepx.tasks import load_task, TaskRunner
```

…or in plain language, via the agent (needs a local LLM):

```bash
sweep-agent chat                   # or: sweepx agent chat
>>> load the Marmousi benchmark model and run a forward — show the shot gather
```

For AI coding assistants, `sweepx skill install` installs the
[Agent Skills](https://sweepx.deepwave.group/sweepx/agent-skills/) ([`skills/sweepx/`](skills/sweepx/)
and the solver's `sweep-solver`) that teach Claude Code or Codex the task schema and the
libraries' APIs; `sweepx skill zip` makes the zips for Claude.ai.

## Development

```bash
git clone https://github.com/DeepWave-KAUST/sweepx.git && cd sweepx
pip install -e ".[dev]"
pytest tests/loss                  # tests/<module>/ for each module
lint-imports                       # the layering contract
```

## License

MIT — see [LICENSE](LICENSE).
