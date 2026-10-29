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
pip install "sweepx[animate]"    # GIF export
pip install "sweepx[segy]"       # SEG-Y files
```

## You also need a local LLM

Chat needs any **OpenAI-compatible** endpoint with a **32k-token context**: the
agent's prompt with its tool schemas is 9-25k tokens.

- **Ollama** (Mac / CPU): `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, then
  `ollama pull qwen2.5:14b`, and run `sweep-agent chat --model qwen2.5:14b`
  (auto-detected on `:11434`). Without the context setting Ollama cuts the prompt
  short without a word. Tool-calling needs a capable model — the default **Q4
  quants are noticeably weaker**; prefer a `-q8_0` tag, a larger model
  (`qwen2.5:32b`), or a full-precision endpoint.
- **vLLM** (GPU node): vLLM pins its own torch, so install it in an environment of
  its own; on NVIDIA drivers 570 / 575 use the CUDA 12 build:

    ```bash
    pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128
    pip install "transformers>=4.45,<5.0"
    vllm serve Qwen/Qwen2.5-14B-Instruct --max-model-len 32768 \
        --enable-auto-tool-choice --tool-call-parser hermes
    ```

    `examples/agent/start_vllm_qwen7b.sh` wraps this (7B by default,
    `MODEL=Qwen/Qwen2.5-14B-Instruct` for 14B). Full precision is the most reliable
    for tool-calling; 14B needs a 48 GB-class GPU. `sweep-agent chat` finds the
    server on `:8000` / `:8001` and asks it which model it serves.

!!! note "macOS (Apple Silicon)"
    Runs end-to-end on M-series with MPS (CPU 26.7 s → MPS 5.5 s on a
    256×384 / 8-shot / 1500-step demo). Use Ollama for the LLM; `run_forward_sweep`
    defaults to `device="auto"` (MPS → CPU). Do **not** set `SWEEP_BUILD_CUDA` —
    that's the Linux + NVIDIA path; macOS uses sweep's eager torch.
