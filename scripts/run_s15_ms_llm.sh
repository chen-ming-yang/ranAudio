#!/usr/bin/env bash
# S1.5 continuation: unfreeze MiniCPM5-2B and train one epoch on Wenet MS.
# Starts from b_ms_cont encoder/projector. Does not overwrite that directory.
# Log: tmp/s15_ms_llm.log and runs/s15_asr/b_ms_llm/train.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp runs/s15_asr/b_ms_llm

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
OUT="${S15_LLM_OUT:-$ROOT/runs/s15_asr/b_ms_llm}"

echo $$ > tmp/s15_ms_llm.pid
exec "$MINI" ranAudio/train_s15_asr.py \
  --samples "$ROOT/data/wenet_ms.jsonl" \
  --encoder "$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt" \
  --projector "$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt" \
  --train-llm \
  --llm-lr 1e-5 \
  --encoder-lr 1e-5 \
  --lr 1e-4 \
  --epochs 1 \
  --max-batch-dur 32 \
  --out-dir "$OUT"
