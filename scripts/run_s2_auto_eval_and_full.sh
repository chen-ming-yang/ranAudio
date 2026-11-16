#!/usr/bin/env bash
# Auto-pipeline: wait for LoRA mix -> eval LoRA -> full SFT -> eval full -> compare.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_adapter() {
  local path="$1"
  log "WAIT for adapter $path"
  while [ ! -f "$path/adapter_model.safetensors" ]; do
    sleep 30
  done
  # Give the writer a moment to flush tokenizer etc.
  sleep 10
  log "FOUND adapter $path"
}

wait_for_llm() {
  local path="$1"
  log "WAIT for full checkpoint $path"
  while [ ! -f "$path/llm_final.safetensors" ]; do
    sleep 30
  done
  sleep 10
  log "FOUND checkpoint $path"
}

# ---------- 1. LoRA four-phase already running; wait for mix ----------
wait_for_adapter "$ROOT/runs/s2_fc_nofc_lora/mix/adapter_final"

log "EVAL LoRA final"
$MINI ranAudio/eval_s2.py \
  --adapter "$ROOT/runs/s2_fc_nofc_lora/mix/adapter_final" \
  --out-json "$ROOT/tmp/s2_test_eval_lora.json" \
  > "$ROOT/tmp/s2_eval_lora.log" 2>&1
log "DONE LoRA eval -> tmp/s2_test_eval_lora.json"

# ---------- 2. Full SFT ----------
log "START full SFT four-phase"
bash "$ROOT/scripts/run_s2_train_full.sh" > "$ROOT/tmp/s2_train_full.log" 2>&1
log "DONE full SFT"

# ---------- 3. Eval full SFT ----------
wait_for_llm "$ROOT/runs/s2_fc_nofc_full/mix"

log "EVAL full SFT final"
$MINI ranAudio/eval_s2.py \
  --adapter "$ROOT/runs/s2_fc_nofc_full/mix" \
  --out-json "$ROOT/tmp/s2_test_eval_full.json" \
  > "$ROOT/tmp/s2_eval_full.log" 2>&1
log "DONE full eval -> tmp/s2_test_eval_full.json"

# ---------- 4. Compare ----------
log "COMPARE"
$MINI - <<'PY' > "$ROOT/tmp/s2_compare.txt"
import json
from pathlib import Path
lora_path = Path("tmp/s2_test_eval_lora.json")
full_path = Path("tmp/s2_test_eval_full.json")
if not lora_path.exists() or not full_path.exists():
    print("One or both eval files missing")
    raise SystemExit(1)
lora = json.load(open(lora_path))['summary']
full = json.load(open(full_path))['summary']
print("S2 Test Set Comparison")
print("=" * 50)
print(f"{'metric':<20} {'LoRA':>10} {'Full SFT':>10}")
print("-" * 50)
print(f"{'route_acc':<20} {lora['route_acc']:>10.3f} {full['route_acc']:>10.3f}")
print(f"{'fc route_acc':<20} {lora['buckets']['fc']['route_acc']:>10.3f} {full['buckets']['fc']['route_acc']:>10.3f}")
print(f"{'fc cer':<20} {lora['buckets']['fc']['cer']:>10.3f} {full['buckets']['fc']['cer']:>10.3f}")
print(f"{'chat route_acc':<20} {lora['buckets']['chat']['route_acc']:>10.3f} {full['buckets']['chat']['route_acc']:>10.3f}")
print(f"{'chat parrot':<20} {lora['buckets']['chat']['parrot']:>10.3f} {full['buckets']['chat']['parrot']:>10.3f}")
print(f"{'asr route_acc':<20} {lora['buckets']['asr']['route_acc']:>10.3f} {full['buckets']['asr']['route_acc']:>10.3f}")
print(f"{'asr cer':<20} {lora['buckets']['asr']['cer']:>10.3f} {full['buckets']['asr']['cer']:>10.3f}")
print(f"{'asr trap_acc':<20} {lora['buckets']['asr'].get('trap_route_acc', 0.0):>10.3f} {full['buckets']['asr'].get('trap_route_acc', 0.0):>10.3f}")
print("=" * 50)
PY
log "COMPARE saved -> tmp/s2_compare.txt"
cat "$ROOT/tmp/s2_compare.txt"
log "ALL DONE"
