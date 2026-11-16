#!/usr/bin/env bash
# S2 route demo: mix_llmcont LLM + b_ms_llm_cont encoder/projector.
# Log: tmp/s2_demo.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
HOST="${S2_DEMO_HOST:-0.0.0.0}"
PORT="${S2_DEMO_PORT:-7860}"
S15="${S15_DEMO_CKPT:-$ROOT/runs/s15_asr/b_ms_llm_cont}"
S2="${S2_DEMO_CKPT:-$ROOT/runs/s2_fc_nofc_full/mix_llmcont}"

echo $$ > tmp/s2_demo.pid
exec "$MINI" ranAudio/demo_s2.py \
  --host "$HOST" \
  --port "$PORT" \
  --encoder "$S15/encoder_best.pt" \
  --projector "$S15/projector_best.pt" \
  --llm-weights "$S2/llm_final.safetensors" \
  --tokenizer-dir "$S2/tokenizer"
