# sweep-agent

A natural-language front end: a **local** OpenAI-compatible LLM (Ollama,
vLLM, llama.cpp, LM Studio, or any endpoint you point it at) calls ~30
registered tools that build validated `sweepx.tasks` specs, run them and plot
the results. If you are an AI assistant with a shell, you normally do not
need it: write the YAML and run `sweepx tasks run` yourself. The agent's tools
are still handy as plain Python functions (below).

## Commands

| command | does |
|---|---|
| `sweepx agent chat [--url URL] [--model M] [--api-key K] [--max-steps 12] [-q]` | terminal chat; auto-detects Ollama (`:11434`) / vLLM when no URL is given |
| `sweepx agent ui [--port 7860] [--share]` | Gradio web UI with file drag-and-drop (`pip install "sweepx[agent-ui]"`) |
| `sweepx agent serve-llm --model qwen2.5-14b-instruct [--port 8000]` | start a local vLLM server (`pip install "sweepx[vllm]"`) |
| `sweepx agent tools [--json]` | list tools (or their OpenAI tool specs) |

Environment: `SWEEP_AGENT_LLM_URL`, `SWEEP_AGENT_LLM_MODEL`,
`SWEEP_AGENT_LLM_API_KEY` (the CLI flags win). Tool calling needs a capable
model; heavily quantized (Q4) local models drift and mis-call tools — prefer
q8 or full precision. Custom backends subclass `sweepx.agent.llm.BaseLLM`.
`sweepx[mcp]` installs the MCP dependency.

## Intent → tool (what the agent's system prompt maps)

| request | tool(s) |
|---|---|
| forward + show the gather | `run_forward_and_plot` (or `build_forward_spec` → `run_task` → `plot_shot_gather`) |
| animate / snapshot a wavefield | `animate_wavefield`; `build_wavefield_spec` → `run_task` → `plot_wavefield` / `make_wavefield_gif` |
| compare equations' wavefronts | `compare_equation_wavefields` |
| FWI (+ figures) | `run_fwi`; multiscale `run_multiscale_fwi`; advanced: `describe_task_schema` → `build_spec` → `run_task` |
| RTM / LSRTM / any advanced key | `describe_task_schema(task_type, section)` → `build_spec` → `run_task` |
| which equations / models | `list_equations` |
| stability / dispersion check | `check_parameters` |
| look at files | `inspect_file`, `plot_velocity_model`, `plot_velocity_slice`, `plot_segy`, `plot_observed_data`, `compare_shot_gathers`, `plot_wavelet` |
| no data yet | `make_synthetic_model` (`elastic=true` gives vp/vs/rho), `list_benchmark_models`, `get_benchmark_model` |
| solver-only install | `run_forward_sweep` |

## Tools as Python functions

Every tool is registered with a pydantic params model; `invoke(dict)`
validates and returns a `ToolResult(ok, value, error)` without any LLM.

```python
from sweepx.agent.tools import registry

r = registry.get("check_parameters").invoke({"dh": 10.0, "dt": 0.002, "fm": 15.0,
                                              "vp_min": 1500.0, "vp_max": 4500.0})
print(r.ok, {k: r.value[k] for k in list(r.value)[:4]})

r = registry.get("make_synthetic_model").invoke({"kind": "anomaly", "nz": 60, "nx": 100,
                                                 "out_path": "anomaly_vp.npy"})
print(r.ok, r.value["shape"] if r.ok else r.error)

r = registry.get("describe_task_schema").invoke({"task_type": "fwi", "section": "optimizer"})
print(r.ok, str(r.value)[:200])
print(sorted(registry.names())[:8])
```

`build_spec` / `build_forward_spec` / `build_fwi_spec` validate with the same
loader as `sweepx tasks run` and write `<output_dir>/<task_id>.yaml` (default
`./sweep_runs`). `make_synthetic_model` defaults `out_path` to
`/tmp/synthetic_vp.npy`; pass your own path. `check_parameters` reports the
Courant number with the `sqrt(ndim)` factor (`vmax*dt*sqrt(ndim)/dh`, OK below
0.9) and points per wavelength at 2.5×fm (good ≥ 5) — stricter than the
runner's own gate, which is `vmax*dt/dh` (warn > 0.85, refuse > 1.0).
