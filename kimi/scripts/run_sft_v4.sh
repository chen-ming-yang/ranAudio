#!/bin/bash
# Background v4 data + train. Restarts L1 on the v3 adapter when training exits.
set -u
ROOT=/cmy/cmy/ranAudio
OUT="$ROOT/kimi/eval_name_clarify/sft_out"
PY=/cmy/cmy/miniconda3/envs/qwen3-compare/bin/python
L1_PY=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python
V3_ADAPTER="$OUT/qwen35_2b_lora_v3"
LOG="$OUT/run_v4.log"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

stop_l1() {
  pkill -f 'scripts/l1_server.py --host' || true
  sleep 2
  pkill -9 -f 'VLLM::EngineCore' || true
  pkill -9 -f 'glyph_postprocess.py --serve' || true
  sleep 2
}

start_l1() {
  cd "$ROOT/kimi"
  env -u PYTHONPATH nohup "$L1_PY" -u scripts/l1_server.py \
    --host 0.0.0.0 --port 7864 \
    --model /cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B \
    --gpu-mem 0.35 \
    --llm-adapter "$V3_ADAPTER" \
    >> "$OUT/l1_7864.log" 2>&1 &
  log "l1 restarted pid $! adapter v3"
}

log "prepare start"
if ! "$PY" "$ROOT/kimi/scripts/prepare_sft_name_clarify.py" \
  --v4-from "$ROOT/kimi/eval_name_clarify/sft_v3" \
  --out "$ROOT/kimi/eval_name_clarify/sft_v4" \
  --seed 41 >> "$LOG" 2>&1; then
  log "prepare failed"
  exit 1
fi
log "prepare done"

log "stop l1"
stop_l1

log "train start"
train_rc=0
PYTHONPATH=/cmy/cmy/Qwen/transformers-main/src \
  SFT_BATCH=32 SFT_GRAD_ACCUM=1 SFT_EPOCHS=2 \
  SFT_DATA_DIR="$ROOT/kimi/eval_name_clarify/sft_v4" \
  SFT_OUTPUT_DIR="$OUT/qwen35_2b_lora_v4" \
  "$PY" -u "$ROOT/kimi/scripts/sft_name_clarify.py" \
  > "$OUT/train_v4.log" 2>&1 || train_rc=$?
log "train exit $train_rc"

start_l1
exit "$train_rc"
