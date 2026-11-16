#!/usr/bin/env bash
set +o noclobber
set -euo pipefail
LOG=/cmy/cmy/ranAudio/wenetSpeechL/outputs/aed_download.log
PIDF=/cmy/cmy/ranAudio/wenetSpeechL/outputs/aed_download.pid
AED=/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED
PY=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python

if [[ -f "$AED/model.pth.tar" ]]; then
  echo "AED already complete"; exit 0
fi
if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "AED download already running pid=$(cat $PIDF)"; exit 0
fi

# kill stale incomplete downloaders
pkill -f 'snapshot_download.*FireRedASR2-AED' 2>/dev/null || true
sleep 1
: > "$LOG"
nohup "$PY" -u -c '
from modelscope import snapshot_download
print("resume FireRedASR2-AED via modelscope", flush=True)
p = snapshot_download("xukaituo/FireRedASR2-AED", local_dir="/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED")
print("done", p, flush=True)
' >>"$LOG" 2>&1 &
echo $! | tee "$PIDF"
echo "started AED download pid=$(cat $PIDF)"
