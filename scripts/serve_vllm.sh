#!/usr/bin/env bash
# Serve a Hugging Face checkpoint with vLLM for harness rollouts (OpenAI-compatible, server-side tool parsing).
#
#   GPU=0 PORT=8000 TOOL_PARSER=qwen3_xml bash scripts/serve_vllm.sh /path/to/model my-model
#   agentic-harness run --provider vllm --base-url http://127.0.0.1:8000 --model my-model --tasks ... --out ...
#   # exact token ids / logprobs for RL data (client-side chat template):
#   agentic-harness run --provider vllm_completions --base-url http://127.0.0.1:8000 --model my-model --tokenizer /path/to/model ...
#
# TOOL_PARSER must match the model's chat template (vLLM: `vllm serve --help | grep -A3 tool-call-parser`),
# e.g. qwen3_xml / qwen3_coder (Qwen3.5, Qwen3-Coder), hermes (Qwen2.5/Qwen3), llama3_json, mistral.
# On a GPU shared with other jobs, pin memory: MEM_UTIL=0.3 KV_BYTES=6442450944 (the script refuses busy GPUs unless FORCE=1).
set -euo pipefail
CKPT="$1"; NAME="${2:-local}"
GPU="${GPU:?set GPU=<index of an idle GPU>}"
PORT="${PORT:-8000}"
MEM_UTIL="${MEM_UTIL:-0.85}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
MAX_LEN="${MAX_LEN:-32768}"
KV_BYTES="${KV_BYTES:-}"
VLLM="${VLLM:-vllm}"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "${FORCE:-0}" != 1 ] && [ "$used" -gt 2000 ]; then
    echo "GPU $GPU already uses ${used}MiB. Refusing; set FORCE=1 if this is intended." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
EXTRA=()
[ -n "$KV_BYTES" ] && EXTRA+=(--kv-cache-memory-bytes "$KV_BYTES")
exec "$VLLM" serve "$CKPT" --served-model-name "$NAME" --port "$PORT" --host 127.0.0.1 \
    --gpu-memory-utilization "$MEM_UTIL" --max-model-len "$MAX_LEN" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" "${EXTRA[@]}"
