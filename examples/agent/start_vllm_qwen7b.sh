#!/usr/bin/env bash
# Launch a local vLLM server with tool calling for sweep-agent.
#
# vLLM pins its own torch, so give it an environment of its own (not the one
# sweepx runs in). The cu12 build (0.10.x) runs on NVIDIA drivers 570 / 575; the
# cu13 wheels of later vLLM releases fail there with "driver too old":
#   python -m venv vllm-env && . vllm-env/bin/activate
#   pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128
#   pip install "transformers>=4.45,<5.0"   # 5.x drops all_special_tokens_extended
#
# Usage (in that environment):
#   bash examples/agent/start_vllm_qwen7b.sh                 # Qwen2.5-7B, first free port from 8000
#   MODEL=Qwen/Qwen2.5-14B-Instruct bash examples/agent/start_vllm_qwen7b.sh
#
# Models download to $HF_HOME (default ~/.cache/huggingface; 15 GB for 7B,
# 30 GB for 14B): point it at a roomier disk if your home has a small quota.
# GPU_MEM is the share of the GPU vLLM takes (default 0.85); lower it if the same
# GPU also runs the simulations.
#
# Then, in the environment that has sweepx (sweep-agent finds :8000 / :8001 and
# the model the server serves by itself):
#   sweep-agent chat

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_LEN="${MAX_LEN:-32768}"   # the system prompt + tool schemas alone are 9-25k tokens
# Qwen2.5 / DeepSeek-R1-Distill(Qwen) → "hermes"; Llama-3.x → "llama3_json".
PARSER="${PARSER:-hermes}"

# --- pick a free port (start at $PORT or 8000) ------------------------------
PORT="${PORT:-8000}"
while ss -tln 2>/dev/null | grep -q ":${PORT}\b"; do
    echo "[start_vllm] port $PORT busy, trying $((PORT+1))"
    PORT=$((PORT+1))
done

echo "[start_vllm] HF_HOME=${HF_HOME:-~/.cache/huggingface}"
echo "[start_vllm] launching $MODEL on port $PORT (parser=$PARSER, max-model-len=$MAX_LEN)"
echo "[start_vllm] → URL http://localhost:$PORT/v1"

# Optional YaRN rope-scaling to extend context past the model's native 32768.
# Set ROPE_SCALING to a JSON string, e.g.
#   ROPE_SCALING='{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":32768}'
# (Qwen2.5 supports YaRN; mild short-context quality trade-off.) Empty = native.
ROPE_ARGS=()
if [ -n "${ROPE_SCALING:-}" ]; then
    ROPE_ARGS=(--rope-scaling "$ROPE_SCALING")
    echo "[start_vllm] rope-scaling: $ROPE_SCALING"
fi

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" \
    "${ROPE_ARGS[@]}"
