#!/usr/bin/env bash
# Launch Kimi L1 streaming ASR demo (Qwen3-ASR-1.7B via vLLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python}"
PORT="${KIMI_L1_PORT:-7863}"
MODE="${1:-qwen}"   # qwen | fake
cd "$ROOT"
mkdir -p tmp
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
# vLLM/triton needs a C compiler for some CUDA kernels
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

ARGS=(scripts/l1_server.py --host 0.0.0.0 --port "$PORT")
if [[ "$MODE" == "fake" ]]; then
  ARGS+=(--fake)
else
  ARGS+=(
    --model "${KIMI_QWEN_ASR:-/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B}"
    --gpu-mem "${KIMI_L1_GPU_MEM:-0.35}"
    --chunk-size-sec "${KIMI_L1_CHUNK:-1.0}"
    --unfixed-token-num "${KIMI_L1_UNFIXED_TOK:-5}"
  )
fi

echo "[run_l1] $PY ${ARGS[*]}"
mkdir -p "$ROOT/tmp"
# avoid noclobber issues when redirected by callers
exec "$PY" "${ARGS[@]}"
