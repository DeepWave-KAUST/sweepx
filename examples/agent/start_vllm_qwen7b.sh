#!/usr/bin/env bash
# Launch a local vLLM server with tool-calling enabled, hardened against the
# gotchas hit on a workstation (RTX 6000 Ada, driver 575 / CUDA 12.9) and on ibex:
#
#   * flashinfer JIT-compiles a sampling kernel at first run and needs `ninja`
#     + `nvcc` on PATH — we prepend the conda env bin and the system CUDA bin.
#   * port 8000 is often already taken — we auto-bump to the next free port.
#   * vLLM must be the cu12 build (0.10.x) to match driver 575; the cu13 wheels
#     (vllm>=0.11) fail with "driver too old". Install with:
#       pip install "vllm==0.10.2" --extra-index-url https://download.pytorch.org/whl/cu128
#       pip install "transformers>=4.45,<5.0"   # 5.x drops all_special_tokens_extended
#
# Usage (from the vllm-server conda env):
#   conda activate vllm-server
#   bash examples/agent/start_vllm_qwen7b.sh                 # Qwen2.5-7B on first free port from 8000
#   MODEL=Qwen/Qwen2.5-14B-Instruct bash examples/agent/start_vllm_qwen7b.sh
#
# Connect from another shell (in the env that has sweep / sweepx.tasks):
#   export SWEEP_AGENT_LLM_URL=http://localhost:<PORT>/v1
#   export SWEEP_AGENT_LLM_MODEL=<MODEL>
#   sweep-agent chat

set -euo pipefail

# --- env paths --------------------------------------------------------------
export HF_HOME="${HF_HOME:-/home/$USER/.cache/huggingface}"

# Locate the conda env bin (this script's interpreter env) + system CUDA so the
# flashinfer JIT build can find ninja + nvcc.
ENV_BIN="$(dirname "$(command -v python)")"
CUDA_BIN=""
for c in /usr/local/cuda-12.9/bin /usr/local/cuda/bin; do
    [ -x "$c/nvcc" ] && CUDA_BIN="$c" && break
done
export PATH="$ENV_BIN${CUDA_BIN:+:$CUDA_BIN}:$PATH"
[ -n "$CUDA_BIN" ] && export CUDA_HOME="$(dirname "$CUDA_BIN")"

# --- config -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_LEN="${MAX_LEN:-32768}"   # system prompt + tool schemas alone are ~16k tokens
# Qwen2.5 / DeepSeek-R1-Distill(Qwen) → "hermes"; Llama-3.x → "llama3_json".
PARSER="${PARSER:-hermes}"

# --- pick a free port (start at $PORT or 8000) ------------------------------
PORT="${PORT:-8000}"
while ss -tln 2>/dev/null | grep -q ":${PORT}\b"; do
    echo "[start_vllm] port $PORT busy, trying $((PORT+1))"
    PORT=$((PORT+1))
done

echo "[start_vllm] HF_HOME=$HF_HOME"
echo "[start_vllm] PATH has ninja=$(command -v ninja || echo MISSING) nvcc=$(command -v nvcc || echo MISSING)"
echo "[start_vllm] launching $MODEL on port $PORT (parser=$PARSER)"
echo "[start_vllm] → connect with: export SWEEP_AGENT_LLM_URL=http://localhost:$PORT/v1"

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
