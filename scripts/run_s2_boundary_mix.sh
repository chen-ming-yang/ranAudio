#!/usr/bin/env bash
# Boundary continuation: edge TTS for new phrases, then one mix epoch from the
# stop-fixed full checkpoint. Encoder and projector stay frozen.
# Logs: tmp/s2_boundary_mix.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp data/s2_boundary_wavs runs/s2_fc_nofc_full

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
QWEN="${QWEN_PY:-/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python}"
SAMPLES="${S2_BOUNDARY_SAMPLES:-$ROOT/data/s2_audio_boundary.jsonl}"
RESUME="${S2_BOUNDARY_RESUME:-$ROOT/runs/s2_fc_nofc_full/mix}"
OUT="${S2_BOUNDARY_OUT:-$ROOT/runs/s2_fc_nofc_full/mix_boundary}"
ENC="${S2_ENCODER:-$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt}"
PROJ="${S2_PROJECTOR:-$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt}"
LR="${S2_BOUNDARY_LR:-5e-6}"
ASR_WEIGHT="${S2_BOUNDARY_ASR_WEIGHT:-1.0}"
BOUNDARY_REPEAT="${S2_BOUNDARY_REPEAT:-8}"
NEG_REPEAT="${S2_NEG_REPEAT:-2}"

log() { echo "[$(date '+%F %T')] $*"; }

echo $$ > tmp/s2_boundary_mix.pid
log "boundary mix pid=$$ resume=$RESUME lr=$LR asr_weight=$ASR_WEIGHT repeat=$BOUNDARY_REPEAT"

log "START boundary texts"
"$MINI" ranAudio/gen_s2_boundary.py
log "OK boundary texts"

log "START edge tts"
"$QWEN" ranAudio/gen_s2_tts.py \
  --jobs "$ROOT/data/s2_boundary_jobs.jsonl" \
  --wav-dir "$ROOT/data/s2_boundary_wavs" \
  --engine edge \
  --concurrency 8
log "OK edge tts"

log "START assemble"
"$MINI" ranAudio/gen_s2_boundary.py --assemble
log "OK assemble -> $SAMPLES"

log "START train mix_boundary -> $OUT"
"$MINI" ranAudio/train_s2_full.py \
  --samples "$SAMPLES" \
  --encoder "$ENC" \
  --projector "$PROJ" \
  --phase mix \
  --epochs 1 \
  --lr "$LR" \
  --asr-weight "$ASR_WEIGHT" \
  --neg-repeat "$NEG_REPEAT" \
  --boundary-repeat "$BOUNDARY_REPEAT" \
  --resume "$RESUME" \
  --out-dir "$OUT"
log "OK train -> $OUT/llm_final.safetensors"
echo "$OUT/llm_final.safetensors" > tmp/s2_boundary_checkpoint.txt

log "START eval"
"$MINI" ranAudio/eval_s2.py \
  --adapter "$OUT" \
  --out-json "$ROOT/tmp/s2_test_eval_full_mix_boundary.json"
log "DONE eval -> tmp/s2_test_eval_full_mix_boundary.json"
