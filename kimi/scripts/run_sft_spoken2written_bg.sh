#!/usr/bin/env bash
# Wait for CS2W expanded text data, prepare SFT jsonl, then LoRA text SFT (background).
# Does NOT wait for TTS by default (text-only). Never touches test.
# Log: kimi/eval_spoken2written/sft.log
set -euo pipefail
set +o noclobber

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/eval_spoken2written"
EXP="$OUT/expanded"
SFT_DIR="$OUT/sft"
LOG="$OUT/sft.log"
PIDF="$OUT/sft.pid"

PY="${SFT_PY:-/cmy/cmy/miniconda3/envs/qwen3/bin/python}"
MODEL="${SFT_MODEL:-/cmy/cmy/Qwen/Qwen3.5-0.8B}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$OUT/sft_out/qwen35_0.8b_lora}"
WAIT_TTS="${SFT_WAIT_TTS:-0}"          # 1 = also wait for expand_tts pipeline done
WAIT_SEC="${SFT_WAIT_SEC:-30}"
MAX_WAIT="${SFT_MAX_WAIT:-86400}"      # seconds

# Prefer free GPU: if TTS holds cuda:0 hard, set SFT_CUDA=1 (second GPU) when present
export CUDA_VISIBLE_DEVICES="${SFT_CUDA:-${CUDA_VISIBLE_DEVICES:-0}}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_expanded() {
  local t=0
  log "wait expanded train+manifest (max ${MAX_WAIT}s)…"
  while true; do
    if [[ -f "$EXP/manifest.json" && -s "$EXP/train.jsonl" && -s "$EXP/val.jsonl" ]]; then
      local n
      n=$(wc -l < "$EXP/train.jsonl" | tr -d ' ')
      if [[ "$n" -ge 100 ]]; then
        log "expanded ready: train_lines=$n"
        return 0
      fi
    fi
    if [[ "$t" -ge "$MAX_WAIT" ]]; then
      log "ERROR timeout waiting for expanded data"
      return 1
    fi
    sleep "$WAIT_SEC"
    t=$((t + WAIT_SEC))
  done
}

wait_tts_idle() {
  [[ "$WAIT_TTS" == "1" ]] || return 0
  local t=0
  log "SFT_WAIT_TTS=1 → wait expand_tts worker exit…"
  while true; do
    if ! pgrep -f 'scripts/run_cs2w_expand_tts_bg.sh --worker' >/dev/null 2>&1 \
       && ! pgrep -f 'scripts/gen_cs2w_tts.py' >/dev/null 2>&1; then
      if grep -q 'CS2W expand + TTS done' "$OUT/expand_tts.log" 2>/dev/null; then
        log "TTS pipeline done"
        return 0
      fi
      # no worker and no done marker: still ok for text SFT if expand exists
      log "no TTS worker; proceed (text SFT)"
      return 0
    fi
    if [[ "$t" -ge "$MAX_WAIT" ]]; then
      log "ERROR timeout waiting for TTS"
      return 1
    fi
    sleep "$WAIT_SEC"
    t=$((t + WAIT_SEC))
  done
}

run_pipeline() {
  log "=== text SFT spoken2written start ==="
  log "PY=$PY MODEL=$MODEL CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  if [[ ! -x "$PY" ]]; then
    log "ERROR python missing: $PY"
    exit 1
  fi
  if ! "$PY" -c "import peft, transformers, torch; assert hasattr(transformers, 'AutoModelForCausalLM')" 2>/dev/null; then
    log "ERROR env missing peft/transformers; use SFT_PY=.../envs/qwen3/bin/python"
    exit 1
  fi

  wait_expanded
  wait_tts_idle

  log "prepare SFT jsonl…"
  "$PY" scripts/prepare_sft_spoken2written.py \
    --expanded-dir "$EXP" \
    --out-dir "$SFT_DIR" \
    --splits train,val

  log "train LoRA → $OUTPUT_DIR"
  "$PY" scripts/sft_spoken2written.py \
    --model "$MODEL" \
    --data-dir "$SFT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "${SFT_EPOCHS:-2}" \
    --batch-size "${SFT_BATCH:-8}" \
    --grad-accum "${SFT_GRAD_ACCUM:-2}" \
    --lr "${SFT_LR:-2e-4}" \
    --max-seq-len "${SFT_MAX_LEN:-512}" \
    --lora-r "${SFT_LORA_R:-16}" \
    --lora-alpha "${SFT_LORA_ALPHA:-32}"

  log "=== text SFT done ==="
  log "adapter: $OUTPUT_DIR"
  log "data:    $SFT_DIR"
}

if [[ "${1:-}" == "--worker" ]]; then
  echo $$ > "$PIDF"
  run_pipeline >>"$LOG" 2>&1
  exit 0
fi

nohup setsid bash "$0" --worker </dev/null >>"$LOG" 2>&1 &
BG=$!
echo "$BG" > "$PIDF"
log "background pid=$BG log=$LOG"
log "tail -f $LOG"
echo "$BG"
