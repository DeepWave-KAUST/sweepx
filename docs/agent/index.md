# sweep-agent

**Offline-LLM natural-language control for the [`sweep`](../solver/) stack** — say
what you want in plain language and a **local** LLM turns it into a validated
`sweep` run. No cloud, no API keys.

> *"here is `vp_init.npy` and `obs.segy`, run an FWI starting at 10 Hz"* → a local
> LLM turns that into a validated `sweep` task and runs it.

Installed with `pip install sweep-agent` (also bundled by `pip install sweepx`)
→ `import sweep_agent`.

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
   tools/  ── inspect_file · list_equations · make_synthetic_model · get_benchmark_model
            · run_forward_sweep · build_fwi_spec · run_task · plot_* · ...
            │
            ├─ discovery / modelling  ──►  sweep                    (core solver)
            └─ build + execute + viz  ──►  sweep_tasks.TaskRunner   (production runner)
```

Tools import the geophysics stack **lazily** — a missing layer returns a clear
`{"error": "… not importable"}` instead of crashing, so the agent always starts.

## What works at each layer

| tools | `pip install sweep-agent` | `+ sweep-tasks` |
|---|:--:|:--:|
| `tools`, `inspect_file`, `check_parameters`, `make_synthetic_model` | ✅ | ✅ |
| `plot_*`, `compare_shot_gathers`, `list_equations` | ✅ | ✅ |
| `run_forward_sweep` — forward modelling (acoustic **+ elastic**) | ✅ | ✅ |
| `list_benchmark_models` / `get_benchmark_model` — Marmousi / Overthrust | ✅ | ✅ |
| `build_*_spec`, `run_task`, `run_fwi`, `run_multiscale_fwi`, … | error dict | ✅ |

Full FWI / LSRTM is the [`sweep-tasks`](../tasks/) tier (the production runner).

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __[Getting started](getting-started/installation.md)__

    ---

    Install, point it at a local LLM, run your first chat.

-   :material-book-open-variant-outline: __[User guide](user-guide/index.md)__

    ---

    Tools, install tiers, and LLM backends (Ollama / vLLM / any OpenAI-compatible).

-   :material-notebook-outline: __[Examples](examples/index.md)__

    ---

    Prompts that work on the base install.

-   :material-api: __[API reference](api/index.md)__

    ---

    The tool functions and the `BaseLLM` backend interface.

</div>
