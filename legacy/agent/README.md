# sweep-agent

Offline-LLM natural-language control for the `sweep` stack.

> Goal: let a user say *"here is `vp_init.npy` and `obs.segy`, run an FWI starting at 10 Hz"* and have a local LLM
> translate that into a validated `sweep_tasks.FWISpec` and execute it via `TaskRunner`.

## Architecture

```
user (NL + files)
      │
      ▼
┌─────────────────────────┐
│  Agent loop (agent.py)  │  ──►  LLM backend (vLLM HTTP) ──► tool_call
│                         │  ◄──  observation (tool result)
└──────────┬──────────────┘
           │ tool_call
           ▼
  ┌──────────────────────────────────────────────┐
  │ tools/                                       │
  │  • inspect_file(path)        ─ peek headers  │
  │  • list_zoo_datasets()       ─ sweep-zoo     │
  │  • build_forward_spec(...)   ─ ForwardSpec   │
  │  • build_fwi_spec(...)       ─ FWISpec       │
  │  • run_task(spec)            ─ TaskRunner    │
  │  • read_status(task_dir)                     │
  └──────────────────────────────────────────────┘
                              │
                              ▼
                  sweep_tasks.TaskRunner.run()
```

## Quickstart

### 1. One-time setup

```bash
# Client side (this repo) — install in the env where you'll run sweep tasks.
pip install -e .

# Server side — keep vLLM in its own conda env to avoid clashing with sweep's
# pinned torch/cuda versions.
conda create -n vllm-server python=3.12 -y
conda activate vllm-server
pip install vllm

# Where downloaded HF models live (default ~/.cache/huggingface is fine; pick a
# disk with plenty of room).
export HF_HOME=/home/$USER/.cache/huggingface
hf download Qwen/Qwen2.5-7B-Instruct          # ~15 GB, first time only
```

### 2. Launch the LLM (on a GPU node, one terminal)

```bash
conda activate vllm-server
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes
```

### 3. Chat (another terminal)

```bash
conda activate ifwitorch         # or whatever env has sweep / sweep_tasks
export SWEEP_AGENT_LLM_URL=http://localhost:8000/v1
export SWEEP_AGENT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
sweep-agent chat
> 帮我对 vp_init.npy 和 obs.segy 做一个 FWI
```

For a non-interactive smoke test that drives one full prompt → tool chain → reply:

```bash
python examples/demo_real_llm.py
```

## Status

Phase 1: skeleton + vLLM client + `inspect_file` tool + smoke test. Phases 2-7 add the FWI/forward
builders, sweep-zoo, MCP server, Jupyter widget, and a Gradio web UI. See top-level project
discussion for the full plan.
