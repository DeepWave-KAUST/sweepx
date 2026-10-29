# User guide

## What you get

`pip install sweepx` brings the `sweep-agent` CLI, the ~30-tool registry, the core
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

## Tools as functions

Every tool is a plain callable (`.fn`) with a pydantic params model, so the whole
toolset is usable from scripts and tests without an LLM:

```python
from sweepx.agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))
```

See the [API reference](../api/index.md) for the full list.
