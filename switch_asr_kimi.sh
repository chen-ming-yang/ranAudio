#!/usr/bin/env bash
# Switch GPU between daytime rest90 ASR (Qwen→FireRed) and nighttime Kimi demo.
#
# Usage:
#   bash switch_asr_kimi.sh asr      # 白天：停 Kimi，续跑 rest90（可 resume）
#   bash switch_asr_kimi.sh kimi     # 晚上：停 ASR，启 Kimi 网页
#   bash switch_asr_kimi.sh status   # 看当前谁在占卡
#   bash switch_asr_kimi.sh stop     # 两边都停
#
# Aliases: day=asr, night=kimi, qwen=asr
set -euo pipefail
set +o noclobber

RANAUDIO="$(cd "$(dirname "$0")" && pwd)"
WENET="$RANAUDIO/wenetSpeechL"
KIMI="$RANAUDIO/kimi"
MODE="${1:-status}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-120}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

alive_pidfile() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local p
  p=$(cat "$f" 2>/dev/null || true)
  [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null
}

kill_pid_tree() {
  local pid="$1"
  [[ -n "${pid:-}" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local c
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_pid_tree "$c"
  done
  kill "$pid" 2>/dev/null || true
}

wait_gone() {
  local pattern="$1"
  local deadline=$((SECONDS + GPU_WAIT_SEC))
  while (( SECONDS < deadline )); do
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  # last resort
  pkill -9 -f "$pattern" 2>/dev/null || true
  sleep 1
}

stop_asr() {
  log "stop rest90 ASR…"
  if [[ -x "$WENET/launch_rest90.sh" ]]; then
    bash "$WENET/launch_rest90.sh" stop || true
  fi
  # pipeline bash + child python (resume-safe; shards keep progress)
  for f in \
    "$WENET/outputs/pipeline_L90pct.pid" \
    "$WENET/outputs/qwen3_asr_L90pct/run.pid" \
    "$WENET/outputs/firered_asr_L90pct/run.pid"
  do
    if alive_pidfile "$f"; then
      kill_pid_tree "$(cat "$f")"
    fi
  done
  pkill -f 'run_qwen3_asr_compare.py.*qwen3_asr_L90pct' 2>/dev/null || true
  pkill -f 'run_firered_asr_compare.py.*firered_asr_L90pct' 2>/dev/null || true
  wait_gone 'run_qwen3_asr_compare.py|run_firered_asr_compare.py|pipeline_L90pct'
  log "ASR stopped (shards kept; resume with: bash switch_asr_kimi.sh asr)"
}

stop_kimi() {
  log "stop Kimi demo…"
  if alive_pidfile "$KIMI/demo.pid"; then
    kill_pid_tree "$(cat "$KIMI/demo.pid")"
  fi
  pkill -f 'scripts/demo_web.py' 2>/dev/null || true
  fuser -k 7861/tcp 7862/tcp 2>/dev/null || true
  wait_gone 'scripts/demo_web.py'
  log "Kimi stopped"
}

wait_gpu_quiet() {
  # best-effort: wait until neither asr nor kimi python holds the GPU
  local deadline=$((SECONDS + GPU_WAIT_SEC))
  while (( SECONDS < deadline )); do
    if ! pgrep -f 'run_qwen3_asr_compare.py|run_firered_asr_compare.py|scripts/demo_web.py' >/dev/null 2>&1; then
      sleep 2
      return 0
    fi
    sleep 2
  done
  log "warn: GPU wait timed out after ${GPU_WAIT_SEC}s; continuing anyway"
}

start_asr() {
  log "start rest90 ASR (resume)…"
  cd "$WENET"
  bash launch_rest90.sh all
  log "ASR up. status: bash $WENET/launch_rest90.sh status"
  log "log:     tail -f $WENET/outputs/pipeline_L90pct.log"
}

start_kimi() {
  log "start Kimi demo…"
  cd "$KIMI"
  : >>"$KIMI/demo.log"
  nohup setsid bash scripts/run_demo.sh >>"$KIMI/demo.log" 2>&1 < /dev/null &
  local bg=$!
  sleep 2
  if ! pgrep -f 'scripts/demo_web.py' >/dev/null 2>&1; then
    log "error: Kimi failed to start; see $KIMI/demo.log"
    return 1
  fi
  log "Kimi launching (pid≈$bg). SenseVoice ready in ~20s; LLM lazy on edit."
  log "open via Cursor Ports → http://localhost:7861  (see kimi/OPEN.md)"
  bash "$KIMI/scripts/open_in_cursor.sh" || log "open skipped; manual: Ports → 7861 → http://localhost:7861"
}

show_status() {
  echo "=== switch_asr_kimi status ==="
  if pgrep -f 'run_qwen3_asr_compare.py|run_firered_asr_compare.py' >/dev/null 2>&1 \
     || alive_pidfile "$WENET/outputs/pipeline_L90pct.pid"; then
    echo "mode: ASR (rest90) RUNNING"
  elif pgrep -f 'scripts/demo_web.py' >/dev/null 2>&1; then
    echo "mode: Kimi demo RUNNING"
  else
    echo "mode: idle (neither)"
  fi
  echo
  bash "$WENET/launch_rest90.sh" status || true
  echo
  if pgrep -f 'scripts/demo_web.py' >/dev/null 2>&1; then
    echo "kimi: RUNNING pid(s)=$(pgrep -f 'scripts/demo_web.py' | tr '\n' ' ')"
    curl -s -m 2 http://127.0.0.1:7861/api/ready 2>/dev/null || echo "(api not ready yet)"
    echo
  else
    echo "kimi: stopped"
  fi
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
}

case "$MODE" in
  asr|day|qwen)
    stop_kimi
    wait_gpu_quiet
    start_asr
    ;;
  kimi|night)
    stop_asr
    wait_gpu_quiet
    start_kimi
    ;;
  stop)
    stop_kimi
    stop_asr
    wait_gpu_quiet
    log "both stopped"
    ;;
  status)
    show_status
    ;;
  *)
    cat <<EOF
Usage: $0 {asr|kimi|status|stop}
  asr|day|qwen  白天：停 Kimi，续跑 rest90 ASR
  kimi|night    晚上：停 ASR，启 Kimi，并用 Cursor Ports 打开 localhost:7861
  status        当前状态
  stop          两边都停
EOF
    exit 1
    ;;
esac
