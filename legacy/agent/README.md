# sweep-agent

Offline-LLM natural-language control for the [`sweep`](https://github.com/GeophyAI) stack.

> **Goal:** say *"here is `vp_init.npy` and `obs.segy`, run an FWI starting at 10 Hz"* and have a
> **local** LLM turn that into a validated `sweep` task and run it — no cloud, no API keys.

## How it works

```
user (natural language + files)
        │
        ▼
┌──────────────────────────┐     tool_call
│  Agent loop (agent.py)   │ ───────────────►  local LLM (OpenAI-compatible: vLLM / Ollama)
│                          │ ◄───────────────  tool result (observation)
└───────────┬──────────────┘
            │  dispatches to one of ~30 registered tools
            ▼
   tools/  ── inspect_file · list_equations · check_parameters · make_synthetic_model
            · build_forward_spec · build_fwi_spec · run_task · plot_* · run_fwi · ...
            │
            ├─ discovery / modelling  ──►  sweep            (core wave-equation solver)
            └─ build + execute + viz  ──►  sweep_tasks.TaskRunner   (production runner)
```

Tools that need the geophysics stack import it **lazily**: if `sweep` / `sweep_tasks` is not
installed, the tool returns a clear `{"error": "... not importable"}` instead of crashing, so
the agent always starts and the pure-Python tools always work.

## Install

sweep-agent installs in **layers**. The base layer is pure-PyPI and runs anywhere
(Linux / macOS, CPU, no GPU); you add the solver, the runner, and an LLM backend as you need them.

### 1. Base — natural-language layer, file inspection, tool registry

```bash
git clone https://github.com/GeophyAI/sweep-agent
cd sweep-agent
pip install -e .          # PyPI deps only: pydantic, httpx, PyYAML, numpy
```

This alone gives you the CLI, the full 30-tool registry, and every tool that doesn't need a
solver — `inspect_file`, `check_parameters`, `make_synthetic_model`, plus the three plotting
tools that only need matplotlib (`plot_wavelet`, `plot_velocity_slice`, `compare_shot_gathers`).
Verify:

```bash
sweep-agent tools        # prints the 30 tools and what each does
```

### 2. The solver — `sweep` (equation discovery + wave modelling)

`sweep` (the core wave-equation solver) is open source but **not on PyPI**. Install your backend
framework **first** — sweep deliberately does not pull PyTorch/JAX for you, so that you control the
CUDA build — then install sweep from the repo root:

```bash
# 1. a working PyTorch (or JAX) environment — see pytorch.org / the JAX install docs
# 2. then:
git clone https://github.com/DeepWave-KAUST/sweep
cd sweep
pip install .
```

That one command covers **both** the PyTorch-eager and the JAX paths — sweep uses lazy imports, so
you only need the framework you actually use.

Note: `sweep`'s default branch is `dev`, so a plain `git clone` checks out the development head.
Pin a tag or commit if you need a reproducible environment.

Optionally, on **Linux + NVIDIA only**, you can additionally build the compiled C++/CUDA binding
(`sweep._C`). sweep-agent's tools do not require it:

```bash
SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
```

Naming: the distribution is **`sweep-solver`** — that is what `pip list` and dependency errors call
it — while the import name is **`sweep`**. Same package.

Verify with `sweep list equations`. This enables `list_equations` and the wave-modelling tools.

### 3. The production runner — `sweep-tasks` (optional, not yet released)

The `build_*_spec`, `run_task`, and `plot_*` tools drive `sweep_tasks.TaskRunner` — the production
task layer of the sweep stack (spec schemas, losses, optimizers, multi-GPU, IO, plotting). It is
**not publicly released yet**, so for now those tools are unavailable outside our group.

That is a soft limit, not a wall: every one of them imports `sweep_tasks` lazily and returns a
plain `{"error": "sweep_tasks is not importable"}` when it is missing, so the agent still starts,
still registers all 30 tools, and everything in steps 1–2 keeps working.

### 4. An LLM backend — to actually *chat*

The agent talks to any **OpenAI-compatible** endpoint. Pick one:

- **GPU node — vLLM:**
  ```bash
  pip install -e '.[vllm]'
  sweep-agent serve-llm --model qwen2.5-14b-instruct     # wraps `vllm serve` with tool-calling on
  ```
- **Mac / CPU — Ollama:** `ollama serve` then `ollama pull qwen2.5:7b`, and point the agent at
  `http://localhost:11434/v1` (see env vars below).

You do **not** need an LLM to develop or unit-test tools — call them directly (see Usage) or list
them with `sweep-agent tools`.

### Optional extras

```bash
pip install -e '.[ui]'       # Gradio web UI (drag-drop files, inline figures)
pip install -e '.[animate]'  # imageio — GIF writing for animate_fwi_evolution
pip install -e '.[test]'     # pytest
pip install -e '.[dev]'      # pytest + ruff + black
```

### macOS (Apple Silicon) — end-to-end

Verified on macOS 15.7 / Apple M3 Max / Python 3.11. The whole chat → tool → modelling → figure
loop runs on an M-series Mac, with MPS acceleration; only large production runs still want a CUDA box.

```bash
# 1. Python 3.9+ (what the package requires); a venv or conda env keeps it isolated
python3 -m venv .venv && source .venv/bin/activate

# 2. Base layer — all deps ship arm64 wheels, so no compiler is needed
pip install -e .
sweep-agent tools                      # should list 30 tools

# 3. The solver. Install PyTorch first (its arm64 wheels give you MPS), then sweep itself.
#    Do NOT set SWEEP_BUILD_CUDA — that is the Linux+NVIDIA compiled-binding path.
pip install torch
git clone https://github.com/DeepWave-KAUST/sweep
(cd sweep && pip install .)

# 4. LLM backend — vLLM is CUDA-only, so use Ollama (native, Metal-accelerated)
brew install ollama
brew services start ollama             # background service; `ollama serve` ties up a terminal
ollama pull qwen2.5:7b                 # 4.7 GB; must be a tool-calling capable model
export SWEEP_AGENT_LLM_URL=http://localhost:11434/v1
export SWEEP_AGENT_LLM_MODEL=qwen2.5:7b
sweep-agent chat
```

Mac notes:

- **Say "mps", not "GPU".** This is the one real trap. Ask the agent to *"run it on the GPU"* and the
  LLM fills `device="cuda"`; the spec fails, the retry silently falls back to `cpu`, and the task
  still reports `state=success` — so you cannot tell it ran on the CPU. Say *"on the mps device"*
  and it sets `device=mps` correctly.
- **MPS is worth it:** 256×384 grid, 8 shots, 1500 steps, eager backend —
  **CPU 26.7 s → MPS 5.5 s (≈4.9× faster)**.
- **Do not** set `SWEEP_BUILD_CUDA=1` — that is the Linux + NVIDIA build path and will fail here.
- macOS runs `sweep`'s **eager torch** path; the compiled `sweep._C` backend is Linux/CUDA-only.
- Ollama's OpenAI-compatible endpoint does real tool calling: `qwen2.5:7b` drives the full
  `inspect_file → build_forward_spec → run_task → plot_shot_gather` chain end to end. It does
  occasionally *say* it will call a tool without actually calling it — a nudge ("plot it") fixes
  that, and `qwen2.5:14b` (~9 GB) is noticeably more reliable if you have the RAM.
- The tool layer needs **neither an LLM nor a GPU**: `sweep-agent tools` and direct `.fn` calls
  (see Usage) work on a bare Mac with just the base install.

## Usage

### Inspect the tools (no LLM, no GPU)

```bash
sweep-agent tools          # human-readable
sweep-agent tools --json   # OpenAI tool-call specs (for wiring into other frameworks)
```

### Chat (needs an LLM endpoint from step 4)

```bash
export SWEEP_AGENT_LLM_URL=http://localhost:8000/v1        # vLLM; or …:11434/v1 for Ollama
export SWEEP_AGENT_LLM_MODEL=qwen2.5-14b-instruct
sweep-agent chat
>>> make a smooth 2-D model and show me how the wave propagates
>>> :reset                 # clear conversation history
```

`--url` / `--model` flags override the env vars; `SWEEP_AGENT_MAX_STEPS` caps tool-call rounds per turn.

### Web UI

```bash
pip install -e '.[ui]'
sweep-agent ui --port 7860     # then open http://localhost:7860
```

Note: the web UI holds **one globally shared conversation** — every open browser window sees the
same history and each other's messages. Use the CLI (`sweep-agent chat`) if you want isolated sessions.

### Call a tool directly from Python (no LLM)

Every tool is a plain function reachable as `.fn`, with a pydantic params model — handy for
scripting and tests:

```python
from sweep_agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))

from sweep_agent.tools.analysis import check_parameters, CheckParametersParams
print(check_parameters.fn(CheckParametersParams(dh=10, dt=1e-3, fm=8, vp_min=1500, vp_max=4500)))
```

## What works at each layer

| tools | base (`pip install -e .`) | + `sweep` | + `sweep-tasks`<br>*(unreleased)* |
|---|:--:|:--:|:--:|
| `sweep-agent tools`, `inspect_file`, `check_parameters`, `make_synthetic_model` | ✅ | ✅ | ✅ |
| `plot_wavelet`, `plot_velocity_slice`, `compare_shot_gathers` *(matplotlib only)* | ✅ | ✅ | ✅ |
| `list_equations` | error dict | ✅ | ✅ |
| `build_*_spec`, `run_task`, the other `plot_*`, `run_fwi`, `run_multiscale_fwi`, … | error dict | error dict | ✅ |

A tool whose layer is missing returns `{"error": "… is not importable"}` — the agent stays up.
Steps 1–2 are all publicly available today; the third column is our internal production tier.

## Tests

```bash
pip install -e '.[test]'
pytest                     # tests that need sweep / sweep_tasks auto-skip when the stack is absent
```

## License

MIT © Shaowen Wang. See `pyproject.toml` for details.
