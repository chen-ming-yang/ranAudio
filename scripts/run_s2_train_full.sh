#!/usr/bin/env bash
# Four-phase S2 full fine-tune. Assumes data/s2_audio.jsonl is ready.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp runs/s2_fc_nofc_full

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
SAMPLES="${S2_SAMPLES:-$ROOT/data/s2_audio.jsonl}"
# Always use dedicated full-SFT dir; ignore inherited S2_RUN_DIR from LoRA runs.
RUN="${S2_FULL_RUN_DIR:-$ROOT/runs/s2_fc_nofc_full}"
ENC="${S2_ENCODER:-$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt}"
PROJ="${S2_PROJECTOR:-$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt}"
EPOCHS="${S2_EPOCHS:-1}"
LR="${S2_LR:-1e-5}"
NEG_REPEAT="${S2_NEG_REPEAT:-2}"

log() { echo "[$(date '+%F %T')] $*"; }

train_phase() {
  local phase="$1" out="$2" resume="${3:-}"
  local extra=()
  if [[ -n "$resume" ]]; then
    extra+=(--resume "$resume")
  fi
  log "START train full phase=$phase -> $out"
  "$MINI" ranAudio/train_s2_full.py \
    --samples "$SAMPLES" \
    --encoder "$ENC" \
    --projector "$PROJ" \
    --phase "$phase" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --neg-repeat "$NEG_REPEAT" \
    --out-dir "$out" \
    "${extra[@]}"
  log "OK train full phase=$phase"
}

ln -sfn ../s15_asr/b_ms_cont/encoder_best.pt "$RUN/encoder.pt"
ln -sfn ../s15_asr/b_ms_cont/projector_best.pt "$RUN/projector.pt"

train_phase route "$RUN/route"
train_phase chat  "$RUN/chat"  "$RUN/route"
train_phase asr   "$RUN/asr"   "$RUN/chat"
train_phase mix   "$RUN/mix"   "$RUN/asr"

log "DONE all full phases -> $RUN/mix/llm_final.safetensors"
echo "$RUN/mix/llm_final.safetensors" > tmp/s2_train_full_checkpoint.txt
