# Installation

```bash
pip install sweep-agent          # the agent + the sweep solver
```

…or as part of the whole umbrella:

```bash
pip install sweepx               # sweep-solver + sweep-agent (+ future companions)
```

Either path installs the `sweep-agent` CLI, the tool registry, and the core
solver (`sweep`). Natural-language **forward modelling and shot gathers work out
of the box** — acoustic **and elastic** (vp/vs/rho), including bundled
**benchmark models** (Marmousi, Overthrust). Python 3.9+.

Extras:

```bash
pip install "sweep-agent[ui]"       # Gradio web UI
pip install "sweep-agent[vllm]"     # vLLM backend (GPU)
pip install "sweep-agent[animate]"  # GIF export
```

## You also need a local LLM

Chat needs any **OpenAI-compatible** endpoint:

- **Ollama** (Mac / CPU): `ollama serve`, then `ollama pull qwen2.5:14b`. Run
  `sweep-agent chat --model qwen2.5:14b` (auto-detected on `:11434`). Tool-calling
  needs a capable model — the default **Q4 quants are noticeably weaker**; prefer a
  `-q8_0` tag, a larger model (`qwen2.5:32b`), or a full-precision endpoint.
- **vLLM** (GPU node): `pip install "sweep-agent[vllm]"`, then
  `sweep-agent serve-llm --model qwen2.5-14b-instruct`. Full-precision — the most
  reliable for tool-calling.

Full **FWI / LSRTM** additionally needs `sweep-tasks` (the production runner) —
**not on PyPI yet**, install from source. Forward modelling and inspection tools
don't need it; an FWI tool called without it returns a clean
`{"error": "sweep_tasks is not importable"}`.

!!! note "macOS (Apple Silicon)"
    Runs end-to-end on M-series with MPS (CPU 26.7 s → MPS 5.5 s on a
    256×384 / 8-shot / 1500-step demo). Use Ollama for the LLM; `run_forward_sweep`
    defaults to `device="auto"` (MPS → CPU). Do **not** set `SWEEP_BUILD_CUDA` —
    that's the Linux + NVIDIA path; macOS uses sweep's eager torch.
