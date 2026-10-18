#!/usr/bin/env bash
# Launch a local vLLM server on a single 48GB-class GPU.
#
# Default: Qwen2.5-14B-Instruct, FP16, tool-calling enabled, 8K context.
# Override the model with the first positional arg:
#   ./serve_vllm_14b.sh deepseek-r1-distill-14b
#
# After the server is up:
#   export SWEEP_AGENT_LLM_URL=http://localhost:8000/v1
#   sweep-agent chat
set -euo pipefail

MODEL_ALIAS="${1:-qwen2.5-14b-instruct}"
PORT="${PORT:-8000}"

# Optional: pin a single GPU (set CUDA_VISIBLE_DEVICES before invoking).
exec sweep-agent serve-llm \
    --model "$MODEL_ALIAS" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype auto
