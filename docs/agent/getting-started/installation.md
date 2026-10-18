# Installation

```bash
pip install sweepx
```

`sweepx.agent` is part of `sweepx`. The install puts the `sweep-agent` command on
your PATH (also reachable as `sweepx agent`) and brings the tool registry, the
solver (`sweep`) and the production runner (`sweepx.tasks`), so natural-language
**forward modelling, shot gathers and FWI / LSRTM all work out of the box** —
acoustic **and elastic** (vp/vs/rho), including bundled **benchmark models**
(Marmousi, Overthrust). Python 3.10+.

Extras:

```bash
pip install "sweepx[agent-ui]"   # Gradio web UI
pip install "sweepx[vllm]"       # vLLM backend (GPU)
pip install "sweepx[animate]"    # GIF export
```

## You also need a local LLM

Chat needs any **OpenAI-compatible** endpoint:

- **Ollama** (Mac / CPU): `ollama serve`, then `ollama pull qwen2.5:14b`. Run
  `sweep-agent chat --model qwen2.5:14b` (auto-detected on `:11434`). Tool-calling
  needs a capable model — the default **Q4 quants are noticeably weaker**; prefer a
  `-q8_0` tag, a larger model (`qwen2.5:32b`), or a full-precision endpoint.
- **vLLM** (GPU node): `pip install "sweepx[vllm]"`, then
  `sweep-agent serve-llm --model qwen2.5-14b-instruct`. Full-precision — the most
  reliable for tool-calling.

!!! note "macOS (Apple Silicon)"
    Runs end-to-end on M-series with MPS (CPU 26.7 s → MPS 5.5 s on a
    256×384 / 8-shot / 1500-step demo). Use Ollama for the LLM; `run_forward_sweep`
    defaults to `device="auto"` (MPS → CPU). Do **not** set `SWEEP_BUILD_CUDA` —
    that's the Linux + NVIDIA path; macOS uses sweep's eager torch.
