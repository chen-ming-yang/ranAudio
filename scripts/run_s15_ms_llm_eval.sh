#!/usr/bin/env bash
# Eval S1.5 b_ms_llm best (unfrozen 2B) on AISHELL-test + car, then test_net.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
CKPT="$ROOT/runs/s15_asr/b_ms_llm"
echo $$ > tmp/s15_ms_llm_eval.pid

echo "=== aishell-test n=200 + car (smoke) ==="
"$MINI" -m ranAudio.eval_asr \
  --split aishell-test \
  --max-samples 200 \
  --also-car \
  --encoder "$CKPT/encoder_best.pt" \
  --projector "$CKPT/projector_best.pt" \
  --llm-weights "$CKPT/llm_best.safetensors" \
  --out-json "$ROOT/tmp/s15_ms_llm_aishell200.json"

echo "=== aishell-test full + car ==="
"$MINI" -m ranAudio.eval_asr \
  --split aishell-test \
  --max-samples 0 \
  --also-car \
  --encoder "$CKPT/encoder_best.pt" \
  --projector "$CKPT/projector_best.pt" \
  --llm-weights "$CKPT/llm_best.safetensors" \
  --out-json "$ROOT/tmp/s15_ms_llm_aishell.json"

echo "=== wenet-test_net full ==="
"$MINI" -m ranAudio.eval_asr \
  --split wenet-test_net \
  --max-samples 0 \
  --encoder "$CKPT/encoder_best.pt" \
  --projector "$CKPT/projector_best.pt" \
  --llm-weights "$CKPT/llm_best.safetensors" \
  --out-json "$ROOT/tmp/s15_ms_llm_test_net.json"

echo "DONE"
