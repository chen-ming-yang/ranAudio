#!/usr/bin/env bash
# S2 first full mix: texts -> (edge TTS ∥ Wenet chat) -> Qwen TTS -> IndexTTS -> assemble.
# Resume-safe. Logs: tmp/s2_build.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp data/s2_tts_wavs data/s2_wenet_chat_wavs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MINI="${MINI_PY:-/cmy/cmy/miniconda3/envs/minicpm5/bin/python}"
QWEN="${QWEN_PY:-/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python}"
INDEX="${INDEX_PY:-/cmy/cmy/miniconda3/envs/indextts/bin/python}"

log() { echo "[$(date '+%F %T')] $*"; }

step() {
  local name="$1"; shift
  log "START $name"
  if "$@"; then
    log "OK $name"
    return 0
  fi
  local rc=$?
  log "FAIL $name rc=$rc"
  return $rc
}

echo $$ > tmp/s2_build.pid
log "S2 build pid=$$ root=$ROOT"

step texts "$MINI" ranAudio/gen_s2_texts.py

# CPU edge in parallel with GPU Wenet.
log "START edge (background)"
"$QWEN" ranAudio/gen_s2_tts.py --engine edge --concurrency 8 \
  > tmp/s2_tts_edge.log 2>&1 &
EDGE_PID=$!
echo "$EDGE_PID" > tmp/s2_tts_edge.pid

WENET_RC=0
step wenet "$MINI" ranAudio/gen_s2_wenet_chat.py --keep 400 --pool 2500 || WENET_RC=$?

log "WAIT edge pid=$EDGE_PID"
wait "$EDGE_PID"
EDGE_RC=$?
log "edge rc=$EDGE_RC (log tmp/s2_tts_edge.log)"

Q_RC=0
for eng in qwen_custom qwen_design qwen_clone; do
  step "tts_$eng" "$QWEN" ranAudio/gen_s2_tts.py --engine "$eng" --device cuda:0 || Q_RC=$?
done

I_RC=0
step tts_index "$INDEX" ranAudio/gen_s2_tts.py --engine index_clone --device cuda:0 || I_RC=$?

A_RC=0
step assemble "$MINI" ranAudio/build_s2_samples.py || A_RC=$?

log "DONE texts ok; wenet=$WENET_RC edge=$EDGE_RC qwen=$Q_RC index=$I_RC assemble=$A_RC"
if [[ "$A_RC" -eq 0 ]]; then
  exit 0
fi
exit 1
