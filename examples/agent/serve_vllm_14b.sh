#!/usr/bin/env bash
# Launch a local vLLM server for sweep-agent on one 48 GB-class GPU.
#
# Default: Qwen2.5-14B-Instruct, FP16, tool calling on, 32K context (the agent's
# prompt with its tool schemas is 9-25k tokens). Override the model with the
# first argument, the rest with environment variables:
#   ./serve_vllm_14b.sh deepseek-r1-distill-14b
#   PORT=8001 GPU_MEM=0.9 MAX_LEN=32768 ./serve_vllm_14b.sh
# Run it in an environment that has both sweepx and vLLM (see
# start_vllm_qwen7b.sh for the vLLM install), or call `vllm serve` the same way
# from a vLLM-only environment.
#
# After the server is up (sweep-agent finds :8000 / :8001 by itself):
#   sweep-agent chat
set -euo pipefail

MODEL_ALIAS="${1:-qwen2.5-14b-instruct}"

# Optional: pin a single GPU (set CUDA_VISIBLE_DEVICES before invoking).
exec sweep-agent serve-llm \
    --model "$MODEL_ALIAS" \
    --port "${PORT:-8000}" \
    --gpu-memory-utilization "${GPU_MEM:-0.85}" \
    --max-model-len "${MAX_LEN:-32768}" \
    --dtype auto
