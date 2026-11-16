#!/usr/bin/env bash
# Projector-only ASR continuation. LLM and encoder stay frozen.
# Logs: tmp/s2_proj_asr.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp runs/s2_fc_nofc_full

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
SAMPLES="${S2_PROJ_SAMPLES:-$ROOT/data/s2_audio_boundary.jsonl}"
RESUME="${S2_PROJ_RESUME:-$ROOT/runs/s2_fc_nofc_full/mix_boundary}"
OUT="${S2_PROJ_OUT:-$ROOT/runs/s2_fc_nofc_full/mix_proj}"
ENC="${S2_ENCODER:-$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt}"
PROJ="${S2_PROJECTOR:-$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt}"
LR="${S2_PROJ_LR:-1e-5}"

log() { echo "[$(date '+%F %T')] $*"; }

echo $$ > tmp/s2_proj_asr.pid
log "proj asr pid=$$ resume=$RESUME projector=$PROJ lr=$LR"

log "START train projector phase=asr -> $OUT"
"$MINI" ranAudio/train_s2_full.py \
  --samples "$SAMPLES" \
  --encoder "$ENC" \
  --projector "$PROJ" \
  --phase asr \
  --epochs 1 \
  --lr "$LR" \
  --asr-weight 1.0 \
  --boundary-repeat 1 \
  --train-projector \
  --freeze-llm \
  --resume "$RESUME" \
  --out-dir "$OUT"
log "OK train -> $OUT/projector_final.pt"
echo "$OUT/projector_final.pt" > tmp/s2_proj_checkpoint.txt

log "START eval"
"$MINI" ranAudio/eval_s2.py \
  --adapter "$OUT" \
  --projector "$OUT/projector_final.pt" \
  --out-json "$ROOT/tmp/s2_test_eval_full_mix_proj.json"
log "DONE eval -> tmp/s2_test_eval_full_mix_proj.json"
