# User guide

## What you get

`pip install sweepx` brings the `sweep-agent` CLI, the 32-tool registry, the core
solver and the production runner (`sweepx.tasks`). Discovery, inspection, plotting,
**forward modelling (acoustic + elastic)**, **benchmark models** and FWI / LSRTM
(`build_*_spec`, `run_task`, `run_fwi`, `run_multiscale_fwi`, …) all work. The
simulations take the GPU when the machine has one — the CUDA kernels for the
equations that have them, eager torch for the rest and for wavefield snapshots —
and the CPU or Apple's MPS otherwise; `device` / `backend_impl` override that.

A tool whose dependencies fail to import returns `{"error": "… is not importable"}`
rather than crashing — the agent stays up, and tools not offered this turn are
rejected (no hallucinated calls). An argument a tool does not take is an error
that names the tool's arguments, so a misspelt one is never dropped silently.

## LLM backends

Chat needs any **OpenAI-compatible** endpoint with a 32k-token context. `chat` /
`ui` are **zero-config by default** (auto-detect Ollama / vLLM, use the model the
server serves; on Ollama pick a 7B and pull it on first run). To switch to
any other backend — a remote vLLM, a hosted endpoint, llama.cpp, LM Studio — pass
`--url` / `--model` / `--api-key`, or set the environment variables:

```bash
export SWEEP_AGENT_LLM_URL=...      # or --url
export SWEEP_AGENT_LLM_MODEL=...    # or --model
export SWEEP_AGENT_LLM_API_KEY=...  # or --api-key
```

For a fully custom backend, subclass `BaseLLM` from `sweepx.agent.llm`.

!!! tip "Model quality matters for tool-calling"
    Heavily-quantized (Q4) local models are weaker at tool-calling and can drift
    language. Prefer `-q8_0`, a larger model, or a full-precision vLLM endpoint.

## MCP clients

`sweep-agent mcp` serves the same tools over the
[Model Context Protocol](https://modelcontextprotocol.io), so an MCP client
(Claude Code, Claude Desktop, Cursor, …) can call them with its own model. The
built-in agent loop and local LLM are not used. Needs the `[mcp]` extra.

```bash
pip install "sweepx[mcp]"
claude mcp add sweepx -- sweep-agent mcp --workdir /path/to/runs
```

For other clients, register the command `sweep-agent` with the arguments
`mcp --workdir /path/to/runs` as a stdio server.

- By default the server offers 18 general tools: `list_equations`,
  `list_benchmark_models`, `get_benchmark_model`, `make_synthetic_model`,
  `inspect_file`, `check_parameters`, `run_forward_sweep`, `describe_task_schema`,
  `build_spec`, `run_task`, `submit_task`, `job_status`, `cancel_job`,
  `read_status`, `list_artifacts`, `plot_model`, `plot_shot_gather` and
  `plot_convergence`. `describe_task_schema` + `build_spec` cover every task type.
  The specialised builders and one-call wrappers are meant for small local models
  and make the tool list about 2.5 times longer; `--tools all` offers them, and
  `--tools name,name,...` offers exactly the tools you name.
- Tools run with the server's working directory set to `--workdir`, so the default
  `./sweep_runs` outputs land there. The server runs where the client starts it.
  Tools such as `run_task` compute on that machine; on a cluster login node, use
  only `submit_task` to run (see below).
- Tools run one at a time, in the server's main thread, so `run_task`'s
  `timeout_s` is enforced.
- Saved figures (PNG/JPEG/GIF up to 4 MB) are returned as images as well as paths,
  so the model can look at them. `--no-images` returns paths only.
- Tool results are sent to the client's model. With a hosted model they leave the
  machine. Use public or synthetic models there, and keep field data on a local
  LLM (`chat` / `ui`).

### Grid spacing recorded with the model

`get_benchmark_model` and `make_synthetic_model` (its `dh` argument, 10 m by
default) save `<stem>.meta.json` next to `<stem>.npy` with the grid spacing. A
tool or `build_spec` call that gets such a model without `dh` uses the recorded
value. A call that gives a different `dh`, or combines models that record
different spacings, is refused. For a `.npy` without the file, `dh` is required.
`run_forward_sweep` returns a `physics` block (record length, source and receiver
positions in metres, Courant number, points per wavelength) with warnings, for
example when the record ends before the direct wave reaches the far receivers.

### `sweep-tasks` commands and Slurm jobs

Four more tools are offered to MCP clients (`chat` / `ui` do not select them;
`sweep_tasks` needs `--tools all` or its name):

| Tool | What it does |
|---|---|
| `sweep_tasks` | Runs `sweep-tasks <argv>` in a subprocess with a timeout (default 600 s) and returns the exit code and output tails. Covers the commands without a dedicated tool: `build-index`, `build-plan`, `extract-coeff`, `analyze-wavelet`, … |
| `submit_task` | Writes an sbatch script that runs `sweep-tasks run` on a task YAML and submits it. Takes the GPU count and type, `nproc_per_node`, `overrides`, time, partition and other `#SBATCH` options, and setup lines. The YAML, with the overrides applied, is validated before submission. |
| `job_status` | Queue state, elapsed time, pending reason or exit code, the log tail and, once the run has started, the task's `status.json`. |
| `cancel_job` | `scancel` for jobs submitted with `submit_task`; other jobs are refused. |

Scripts and Slurm logs go to `./sweep_jobs/<name>-<time>/`, and each job is recorded
in `./sweep_jobs/<jobid>.json`. A spec built on a node without GPUs needs
`device="cpu"` at build time; pass `overrides=["device=cuda"]` to `submit_task`.

## Tools as functions

Every tool is a plain callable (`.fn`) with a pydantic params model, so the whole
toolset is usable from scripts and tests without an LLM:

```python
from sweepx.agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))
```

See the [API reference](../api/index.md) for the full list.
