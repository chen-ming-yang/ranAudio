#!/usr/bin/env bash
# Full background: expand CS2W train/val → multi-engine TTS (edge then qwen_*).
# Never touches test. Log: kimi/eval_spoken2written/expand_tts.log
set -euo pipefail
set +o noclobber

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/eval_spoken2written"
mkdir -p "$OUT" "$OUT/expanded" "$OUT/wavs" tmp

PY="${KIMI_PY:-/cmy/cmy/miniconda3/envs/kimi-audio/bin/python}"
EDGE_PY="${EDGE_PY:-}"
if [[ -z "$EDGE_PY" ]]; then
  if "$PY" -c "import edge_tts" 2>/dev/null; then
    EDGE_PY="$PY"
  elif [[ -x /cmy/cmy/miniconda3/envs/minicpm5/bin/python ]]; then
    EDGE_PY=/cmy/cmy/miniconda3/envs/minicpm5/bin/python
  else
    EDGE_PY="$PY"
  fi
fi
QWEN_PY="${QWEN_PY:-/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python}"
ENGINES="${CS2W_TTS_ENGINES:-edge,qwen_custom,qwen_design}"
DEVICE="${CS2W_TTS_DEVICE:-cuda:0}"
LIMIT="${CS2W_TTS_LIMIT:-0}"
CONC="${CS2W_TTS_CONCURRENCY:-8}"
BATCH="${CS2W_TTS_BATCH:-16}"
LOG="$OUT/expand_tts.log"
PIDF="$OUT/expand_tts.pid"

log() { echo "[$(date '+%F %T')] $*"; }

LIMIT_ARGS=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

run_pipeline() {
  log "=== CS2W expand + TTS start engines=$ENGINES limit=$LIMIT batch=$BATCH ==="
  log "expand…"
  "$PY" scripts/expand_cs2w.py --splits train,val "${LIMIT_ARGS[@]}"

  # edge (CPU/network)
  if [[ ",$ENGINES," == *",edge,"* ]] || [[ "$ENGINES" == "edge" ]]; then
    log "TTS edge via $EDGE_PY"
    "$EDGE_PY" scripts/gen_cs2w_tts.py \
      --splits train,val \
      --engines edge \
      --concurrency "$CONC" \
      "${LIMIT_ARGS[@]}" || log "WARN edge finished with errors"
  fi

  # GPU engines serial (batched for higher util)
  for eng in qwen_custom qwen_design; do
    if [[ ",$ENGINES," == *",$eng,"* ]]; then
      if [[ -x "$QWEN_PY" ]]; then
        log "TTS $eng via $QWEN_PY device=$DEVICE batch=$BATCH"
        "$QWEN_PY" scripts/gen_cs2w_tts.py \
          --splits train,val \
          --engines "$eng" \
          --device "$DEVICE" \
          --batch-size "$BATCH" \
          "${LIMIT_ARGS[@]}" || log "WARN $eng failed (continue)"
      else
        log "WARN skip $eng: QWEN_PY missing ($QWEN_PY)"
      fi
    fi
  done

  log "=== CS2W expand + TTS done ==="
  log "manifest: $OUT/expanded/manifest.json"
  log "index:    $OUT/expanded/tts_index.jsonl"
  log "wavs:     $OUT/wavs/{edge,qwen_custom,qwen_design}/"
}

# If already launched as worker
if [[ "${1:-}" == "--worker" ]]; then
  echo $$ > "$PIDF"
  run_pipeline >>"$LOG" 2>&1
  exit 0
fi

# Detach
nohup setsid bash "$0" --worker </dev/null >>"$LOG" 2>&1 &
BG=$!
echo "$BG" > "$PIDF"
log "background pid=$BG log=$LOG"
log "tail -f $LOG"
echo "$BG"
