# sweep-agent

Offline-LLM natural-language control for the [`sweep`](https://github.com/DeepWave-KAUST/sweep) stack.

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
            · get_benchmark_model · run_forward_sweep · build_fwi_spec · run_task · plot_* · run_fwi · ...
            │
            ├─ discovery / modelling  ──►  sweep            (core wave-equation solver)
            └─ build + execute + viz  ──►  sweep_tasks.TaskRunner   (production runner)
```

Tools import the geophysics stack **lazily**: if a layer is missing, the tool returns a clear
`{"error": "... not importable"}` instead of crashing, so the agent always starts and the tools
that don't need that layer always work.

## Install

```bash
pip install sweep-agent          # the agent + the sweep solver
```

or get it as part of the whole sweep umbrella:

```bash
pip install sweepx               # sweep-solver + sweep-agent (+ future companions)
```

Either path installs the `sweep-agent` CLI, the tool registry, and the core solver
(`sweep-solver`, imports as `sweep`) — so natural-language **forward modelling and shot gathers
work out of the box**, including **elastic** (vp/vs/rho) as well as acoustic, and on **bundled
benchmark models** (Marmousi, Overthrust — `get_benchmark_model`). Python 3.9+.
(Wavefield *animations* and full FWI/LSRTM are the `sweep-tasks` tier — see below.)

**To chat you also need a local LLM** — any OpenAI-compatible endpoint:

- **Ollama** (Mac / CPU): `ollama serve` then `ollama pull qwen2.5:14b`; run `sweep-agent chat --model qwen2.5:14b` (auto-detected). Tool-calling needs a capable model — the default Q4 quants are noticeably weaker; for better local quality use a `-q8_0` tag or a larger model (`qwen2.5:32b`), or point `--url` at a full-precision endpoint.
- **vLLM** (GPU node): `pip install "sweep-agent[vllm]"` then `sweep-agent serve-llm --model qwen2.5-14b-instruct`. Full-precision — the most reliable for tool-calling.

**Full FWI / LSRTM** additionally needs `sweep-tasks` (the production runner: spec schemas, losses,
optimizers, multi-GPU, IO). It is **not on PyPI yet** — install it from source for now. Forward
modelling and the inspection tools don't need it; an FWI tool called without it just returns a clean
`{"error": "sweep_tasks is not importable"}`.

Extras: `pip install "sweep-agent[ui]"` (Gradio web UI), `[vllm]`, `[animate]` (GIF export).

<details>
<summary><b>macOS (Apple Silicon)</b></summary>

Runs end-to-end on M-series with MPS acceleration (CPU 26.7 s → MPS 5.5 s on a 256×384 / 8-shot /
1500-step demo). Use Ollama for the LLM. `run_forward_sweep` defaults to `device="auto"` (picks
MPS → CPU automatically), so you no longer need to name the device. Do **not** set
`SWEEP_BUILD_CUDA` (that's the Linux + NVIDIA compiled-binding path); macOS uses sweep's eager torch.
</details>

## Usage

```bash
sweep-agent chat        # interactive; auto-detects Ollama/vLLM, tells you if none is running
sweep-agent ui          # same agent in a browser (needs [ui] + a running LLM), then open :7860
sweep-agent tools       # list the ~30 tools — no LLM/GPU needed; --json emits OpenAI tool specs
```

```text
$ sweep-agent chat
>>> here is vp_init.npy — run a 2-D acoustic forward and show me the shot gather
>>> load the Marmousi benchmark model and run a forward — show the shot gather
>>> make an elastic model (vp/vs/rho) and run an Elastic forward
>>> :reset              # clear conversation history
```

`chat` / `ui` are **zero-config by default** — they auto-detect a running Ollama (`:11434`) or
vLLM (`:8000`/`:8001`), pick a 7B model, and pull it on first run. **To switch to any other
OpenAI-compatible backend** (a remote vLLM, a hosted endpoint, llama.cpp, LM Studio, …) pass
`--url` / `--model` / `--api-key`, or set `SWEEP_AGENT_LLM_URL` / `SWEEP_AGENT_LLM_MODEL` /
`SWEEP_AGENT_LLM_API_KEY`. For a fully custom backend, subclass `BaseLLM` from `sweep_agent.llm`.

Every tool is also a plain function (`.fn`, with a pydantic params model) — handy for scripts and tests:

```python
from sweep_agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))
```

## What works at each layer

| tools | `pip install sweep-agent` | `+ sweep-tasks`<br>*(from source)* |
|---|:--:|:--:|
| `sweep-agent tools`, `inspect_file`, `check_parameters`, `make_synthetic_model` | ✅ | ✅ |
| `plot_wavelet`, `plot_velocity_slice`, `compare_shot_gathers`, `list_equations` | ✅ | ✅ |
| `run_forward_sweep` — forward modelling (acoustic **+ elastic**) → shot gathers | ✅ | ✅ |
| `list_benchmark_models`, `get_benchmark_model` — load Marmousi / Overthrust / … | ✅ | ✅ |
| `build_*_spec`, `run_task`, other `plot_*`, `run_fwi`, `run_multiscale_fwi`, … | error dict | ✅ |

A tool whose layer is missing returns `{"error": "… is not importable"}` — the agent stays up.
The last column (`sweep-tasks`) is our production FWI/LSRTM tier, not on PyPI yet.

## Tests

```bash
pip install "sweep-agent[test]"
pytest                  # tests that need sweep / sweep_tasks auto-skip when the stack is absent
```

## License

MIT © Shaowen Wang.
