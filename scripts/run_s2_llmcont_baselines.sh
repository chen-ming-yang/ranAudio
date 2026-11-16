#!/usr/bin/env bash
# A/B/C baselines for mix_llmcont experiment. Log: tmp/s2_llmcont_baselines.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp
MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ENC_CONT="$ROOT/runs/s15_asr/b_ms_llm_cont/encoder_best.pt"
PROJ_CONT="$ROOT/runs/s15_asr/b_ms_llm_cont/projector_best.pt"
LLM_CONT="$ROOT/runs/s15_asr/b_ms_llm_cont/llm_best.safetensors"
ENC_BMS="$ROOT/runs/s15_asr/b_ms_cont/encoder_best.pt"
PROJ_BMS="$ROOT/runs/s15_asr/b_ms_cont/projector_best.pt"
S2="$ROOT/runs/s2_fc_nofc_full/mix_boundary"

log() { echo "[$(date '+%F %T')] $*"; }
echo $$ > tmp/s2_llmcont_baselines.pid

log "A_s15 eval_asr aishell-50"
"$MINI" ranAudio/eval_asr.py --split aishell-test --max-samples 50 \
  --encoder "$ENC_CONT" --projector "$PROJ_CONT" --llm-weights "$LLM_CONT" \
  --out-json "$ROOT/tmp/s15_asr_aishell50_A_s15.json"

log "B_s2 eval_asr aishell-50"
"$MINI" ranAudio/eval_asr.py --split aishell-test --max-samples 50 \
  --encoder "$ENC_BMS" --projector "$PROJ_BMS" --adapter "$S2" \
  --out-json "$ROOT/tmp/s15_asr_aishell50_B_s2.json"

log "C_swap eval_asr aishell-50"
"$MINI" ranAudio/eval_asr.py --split aishell-test --max-samples 50 \
  --encoder "$ENC_CONT" --projector "$PROJ_CONT" --adapter "$S2" \
  --out-json "$ROOT/tmp/s15_asr_aishell50_C_swap.json"

log "B_s2 eval_s2"
"$MINI" ranAudio/eval_s2.py --adapter "$S2" \
  --encoder "$ENC_BMS" --projector "$PROJ_BMS" \
  --out-json "$ROOT/tmp/s2_test_eval_B_s2.json"

log "C_swap eval_s2"
"$MINI" ranAudio/eval_s2.py --adapter "$S2" \
  --encoder "$ENC_CONT" --projector "$PROJ_CONT" \
  --out-json "$ROOT/tmp/s2_test_eval_C_swap_llmcont.json"

log "DONE baselines"
