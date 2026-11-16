#!/usr/bin/env bash
# Wait until S2 dataset assemble finishes, then run four-phase LoRA.
# Polls tmp/s2_build.log + data/s2_audio.jsonl. Does not start while GPU TTS is up.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tmp

export PYTHONUNBUFFERED=1
POLL="${S2_WAIT_POLL:-60}"
MIN_N="${S2_TRAIN_MIN_N:-6000}"
GPU_MB="${S2_TRAIN_GPU_FREE_MB:-4096}"

log() { echo "[$(date '+%F %T')] $*"; }

build_running() {
  local pid_file="$ROOT/tmp/s2_build.pid"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

gpu_mem_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0
}

jsonl_n() {
  local meta="$ROOT/data/s2_audio.meta.json"
  if [[ -f "$meta" ]]; then
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('n',0))" "$meta" 2>/dev/null || echo 0
    return
  fi
  if [[ -f "$ROOT/data/s2_audio.jsonl" ]]; then
    wc -l < "$ROOT/data/s2_audio.jsonl" | tr -d ' '
    return
  fi
  echo 0
}

assemble_ok() {
  grep -qE 'DONE .*assemble=0' "$ROOT/tmp/s2_build.log" 2>/dev/null
}

data_ready() {
  assemble_ok || return 1
  [[ -f "$ROOT/data/s2_audio.jsonl" ]] || return 1
  local n
  n="$(jsonl_n)"
  [[ "${n:-0}" -ge "$MIN_N" ]]
}

echo $$ > tmp/s2_wait_train.pid
log "wait-train pid=$$ min_n=$MIN_N poll=${POLL}s"

if [[ -d "$ROOT/runs/s2_fc_nofc/mix/adapter_final" && "${S2_FORCE_TRAIN:-0}" != "1" ]]; then
  log "mix adapter already exists; set S2_FORCE_TRAIN=1 to retrain"
  exit 0
fi

while true; do
  if build_running; then
    log "build still running (edge/qwen/index/assemble); n=$(jsonl_n) gpu=$(gpu_mem_mb)MiB"
    sleep "$POLL"
    continue
  fi
  if data_ready; then
    mem="$(gpu_mem_mb)"
    if [[ "${mem:-0}" -gt "$GPU_MB" ]]; then
      log "data ready n=$(jsonl_n) but GPU ${mem}MiB > ${GPU_MB}; waiting"
      sleep "$POLL"
      continue
    fi
    log "data ready n=$(jsonl_n) gpu=${mem}MiB -> train"
    break
  fi
  if [[ -f "$ROOT/tmp/s2_build.log" ]] && grep -qE 'DONE .*assemble=' "$ROOT/tmp/s2_build.log"; then
    log "build finished but dataset not ready (need assemble=0 and n>=$MIN_N, have n=$(jsonl_n))"
    exit 1
  fi
  log "waiting for assemble; n=$(jsonl_n) gpu=$(gpu_mem_mb)MiB"
  sleep "$POLL"
done

log "hand off to run_s2_train.sh"
exec bash "$ROOT/scripts/run_s2_train.sh" >> "$ROOT/tmp/s2_train.log" 2>&1
