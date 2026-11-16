#!/usr/bin/env bash
# Continue S1.5 unfrozen-2B MS into b_ms_llm_cont.
# Default warm-start: local step940003 (last complete ckpt before crash).
# Override with S15_ENC / S15_PROJ / S15_LLM or S15_LLM_PREV (uses that dir's *_best.*).
# Log: tmp/s15_ms_llm_cont.log and runs/.../train.log
# Launch detached: nohup setsid bash scripts/run_s15_ms_llm_cont.sh >> tmp/s15_ms_llm_cont.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp runs/s15_asr/b_ms_llm_cont

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
OUT="${S15_LLM_OUT:-$ROOT/runs/s15_asr/b_ms_llm_cont}"
PREV="${S15_LLM_PREV:-$OUT}"

ENC="${S15_ENC:-$PREV/encoder_step940003.pt}"
PROJ="${S15_PROJ:-$PREV/projector_step940003.pt}"
LLM="${S15_LLM:-$PREV/llm_step940003.safetensors}"
# fall back to best if step940003 missing
if [[ ! -f "$ENC" || ! -f "$PROJ" || ! -f "$LLM" ]]; then
  ENC="${S15_ENC:-$PREV/encoder_best.pt}"
  PROJ="${S15_PROJ:-$PREV/projector_best.pt}"
  LLM="${S15_LLM:-$PREV/llm_best.safetensors}"
fi

for f in "$ENC" "$PROJ" "$LLM"; do
  if [[ ! -f "$f" ]]; then
    echo "missing weight: $f" >&2
    exit 1
  fi
done

echo $$ > tmp/s15_ms_llm_cont.pid
echo "[$(date -Is)] start pid=$$ enc=$ENC proj=$PROJ llm=$LLM out=$OUT" >> tmp/s15_ms_llm_cont.log

exec "$MINI" -u ranAudio/train_s15_asr.py \
  --samples "$ROOT/data/wenet_ms.jsonl" \
  --encoder "$ENC" \
  --projector "$PROJ" \
  --resume-llm "$LLM" \
  --train-llm \
  --llm-lr 1e-5 \
  --encoder-lr 1e-5 \
  --lr 1e-4 \
  --epochs 1 \
  --max-batch-dur 32 \
  --seed 44 \
  --out-dir "$OUT"
