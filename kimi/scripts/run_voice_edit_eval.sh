#!/usr/bin/env bash
# Voice-edit eval: synthesize missing wavs (edge by default) + score against Kimi demo.
# Log: kimi/tmp/voice_edit_eval.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp eval_voice_edit/reports

PY="${KIMI_PY:-/cmy/cmy/miniconda3/envs/kimi-audio/bin/python}"
# edge-tts: prefer kimi-audio if installed, else minicpm5
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
# qwen engines need qwen3-tts env if available
QWEN_PY="${QWEN_PY:-/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python}"
BASE="${KIMI_DEMO_BASE:-http://127.0.0.1:7861}"
ENGINES="${VOICE_EDIT_ENGINES:-edge}"

log() { echo "[$(date '+%F %T')] $*"; }
echo $$ > tmp/voice_edit_eval.pid

log "oracle-text (rule only)"
"$PY" scripts/eval_voice_edit.py --mode oracle-text \
  --out-json eval_voice_edit/reports/oracle_text.json \
  --out-md eval_voice_edit/reports/oracle_text.md || true

log "TTS engines=$ENGINES"
if [[ "$ENGINES" == *edge* ]]; then
  log "TTS edge via $EDGE_PY"
  "$EDGE_PY" scripts/gen_voice_edit_tts.py --engines edge
fi
# optional GPU TTS with qwen env
for eng in qwen_custom qwen_design; do
  if [[ ",$ENGINES," == *",$eng,"* ]] || [[ "$ENGINES" == "$eng" ]]; then
    if [[ -x "$QWEN_PY" ]]; then
      log "TTS $eng via $QWEN_PY"
      "$QWEN_PY" scripts/gen_voice_edit_tts.py --engines "$eng" || log "WARN $eng failed"
    else
      log "WARN skip $eng: QWEN_PY missing"
    fi
  fi
done

log "HTTP e2e base=$BASE engines=$ENGINES"
"$PY" scripts/eval_voice_edit.py --mode http --base "$BASE" --engines "$ENGINES" \
  --out-json eval_voice_edit/reports/latest.json \
  --out-md eval_voice_edit/reports/latest.md
log "DONE -> eval_voice_edit/reports/latest.md"
