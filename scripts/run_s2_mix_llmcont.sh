#!/usr/bin/env bash
# Resume mix_boundary with b_ms_llm_cont encoder/projector (frozen).
# Output: runs/s2_fc_nofc_full/mix_llmcont
# Logs: tmp/s2_mix_llmcont.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp runs/s2_fc_nofc_full

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
SAMPLES="${S2_MIX_LLMCONT_SAMPLES:-$ROOT/data/s2_audio_boundary.jsonl}"
RESUME="${S2_MIX_LLMCONT_RESUME:-$ROOT/runs/s2_fc_nofc_full/mix_boundary}"
OUT="${S2_MIX_LLMCONT_OUT:-$ROOT/runs/s2_fc_nofc_full/mix_llmcont}"
ENC="${S2_ENCODER:-$ROOT/runs/s15_asr/b_ms_llm_cont/encoder_best.pt}"
PROJ="${S2_PROJECTOR:-$ROOT/runs/s15_asr/b_ms_llm_cont/projector_best.pt}"
LR="${S2_MIX_LLMCONT_LR:-5e-6}"
ASR_WEIGHT="${S2_MIX_LLMCONT_ASR_WEIGHT:-1.0}"
NEG_REPEAT="${S2_NEG_REPEAT:-2}"
BOUNDARY_REPEAT="${S2_BOUNDARY_REPEAT:-8}"

log() { echo "[$(date '+%F %T')] $*"; }

echo $$ > tmp/s2_mix_llmcont.pid
log "mix_llmcont pid=$$ resume=$RESUME enc=$ENC proj=$PROJ lr=$LR asr_weight=$ASR_WEIGHT out=$OUT"

log "START train mix_llmcont -> $OUT"
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
echo "$OUT/llm_final.safetensors" > tmp/s2_mix_llmcont_checkpoint.txt

log "START eval_s2"
"$MINI" ranAudio/eval_s2.py \
  --adapter "$OUT" \
  --encoder "$ENC" \
  --projector "$PROJ" \
  --out-json "$ROOT/tmp/s2_test_eval_mix_llmcont.json"
log "DONE eval_s2 -> tmp/s2_test_eval_mix_llmcont.json"

log "START eval_asr aishell-50"
"$MINI" ranAudio/eval_asr.py \
  --split aishell-test \
  --max-samples 50 \
  --encoder "$ENC" \
  --projector "$PROJ" \
  --adapter "$OUT" \
  --out-json "$ROOT/tmp/s15_asr_aishell50_mix_llmcont.json"
log "DONE eval_asr -> tmp/s15_asr_aishell50_mix_llmcont.json"
