# User guide

## The two tiers

`sweep-agent` degrades gracefully around what's installed:

- **Base install (`pip install sweep-agent`)** — the CLI, the ~30-tool registry,
  and the core solver. Discovery, inspection, plotting, **forward modelling
  (acoustic + elastic)**, and **benchmark models** all work.
- **`+ sweep-tasks`** — unlocks the production tier: `build_*_spec`, `run_task`,
  `run_fwi`, `run_multiscale_fwi`, and the rest of FWI / LSRTM.

A tool whose layer is missing returns `{"error": "… is not importable"}` rather
than crashing — the agent stays up, and tools not offered this turn are rejected
(no hallucinated calls).

## LLM backends

Chat needs any **OpenAI-compatible** endpoint. `chat` / `ui` are **zero-config by
default** (auto-detect Ollama / vLLM, pick a 7B, pull on first run). To switch to
any other backend — a remote vLLM, a hosted endpoint, llama.cpp, LM Studio — pass
`--url` / `--model` / `--api-key`, or set the environment variables:

```bash
export SWEEP_AGENT_LLM_URL=...      # or --url
export SWEEP_AGENT_LLM_MODEL=...    # or --model
export SWEEP_AGENT_LLM_API_KEY=...  # or --api-key
```

For a fully custom backend, subclass `BaseLLM` from `sweep_agent.llm`.

!!! tip "Model quality matters for tool-calling"
    Heavily-quantized (Q4) local models are weaker at tool-calling and can drift
    language. Prefer `-q8_0`, a larger model, or a full-precision vLLM endpoint.

## Tools as functions

Every tool is a plain callable (`.fn`) with a pydantic params model, so the whole
toolset is usable from scripts and tests without an LLM:

```python
from sweep_agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))
```

See the [API reference](../api/index.md) for the full list.
