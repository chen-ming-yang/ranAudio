#!/usr/bin/env bash
# Resume S2 LoRA from chat phase (route already done). Same hyper-params as run_s2_train.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
SAMPLES="${S2_SAMPLES:-$ROOT/data/s2_audio.jsonl}"
RUN="${S2_RUN_DIR:-$ROOT/runs/s2_fc_nofc_lora}"
ENC="${S2_ENCODER:-$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt}"
PROJ="${S2_PROJECTOR:-$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt}"
EPOCHS="${S2_EPOCHS:-1}"
LR="${S2_LR:-2e-4}"
NEG_REPEAT="${S2_NEG_REPEAT:-2}"

log() { echo "[$(date '+%F %T')] $*"; }

train_phase() {
  local phase="$1" out="$2" resume="${3:-}"
  local extra=()
  if [[ -n "$resume" ]]; then
    extra+=(--resume "$resume")
  fi
  log "START train phase=$phase -> $out"
  "$MINI" ranAudio/train_s2_route.py \
    --samples "$SAMPLES" \
    --encoder "$ENC" \
    --projector "$PROJ" \
    --phase "$phase" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --neg-repeat "$NEG_REPEAT" \
    --out-dir "$out" \
    "${extra[@]}"
  log "OK train phase=$phase"
}

train_phase chat "$RUN/chat" "$RUN/route/adapter_final"
train_phase asr  "$RUN/asr"  "$RUN/chat/adapter_final"
train_phase mix  "$RUN/mix"  "$RUN/asr/adapter_final"

log "DONE resume phases -> $RUN/mix/adapter_final"
echo "$RUN/mix/adapter_final" > tmp/s2_train_adapter.txt
