#!/usr/bin/env bash
# One-click pipeline for the remaining ~90% of WenetSpeech L (complement of the 10% Tier-A pool).
# Flow: Qwen3-ASR -> FireRedASR2-AED -> retry Qwen rows that failed to slice -> Tier-A filter.
# The compare step runs only after that Qwen retry (see filter_tier_A.py).
# Supports resume: re-run the same command; finished utt_ids in shard_*.jsonl are skipped.
#
# Usage:
#   bash launch_rest90.sh all          # prepare + serial qwen->firered->filter (bg)  ★真正开跑
#   bash launch_rest90.sh prepare      # only build sample_manifest_rest90.jsonl
#   bash launch_rest90.sh qwen         # only Qwen (bg, resumable)
#   bash launch_rest90.sh firered      # only FireRed (bg, resumable)
#   bash launch_rest90.sh filter       # only Tier-A filter (bg, resumable via re-run)
#   bash launch_rest90.sh status       # print progress (default if no args)
#   bash launch_rest90.sh stop         # kill pipeline / model jobs
#
# Env overrides: PYTHON, QWEN_MODEL, FIRERED_MODEL, DEVICE, QWEN_BS, FIRERED_BS
# H800 defaults: larger batches to raise GPU util (override down if OOM).
set -euo pipefail
set +o noclobber

ROOT="/cmy/cmy/ranAudio/wenetSpeechL"
MODE="${1:-status}"
PYTHON="${PYTHON:-/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python}"
MANIFEST="${MANIFEST:-$ROOT/sample_manifest_rest90.jsonl}"
QWEN_OUT="${QWEN_OUT:-$ROOT/outputs/qwen3_asr_L90pct}"
FIRERED_OUT="${FIRERED_OUT:-$ROOT/outputs/firered_asr_L90pct}"
TIER_A_OUT="${TIER_A_OUT:-$ROOT/outputs/tier_A_agree3_rest90.jsonl}"
TIER_A_SUMMARY="${TIER_A_SUMMARY:-$ROOT/outputs/tier_A_summary_rest90.json}"
QWEN_MODEL="${QWEN_MODEL:-/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B}"
FIRERED_MODEL="${FIRERED_MODEL:-/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED}"
FIRERED_ROOT="${FIRERED_ROOT:-/cmy/cmy/FireRedASR2S}"
DEVICE="${DEVICE:-cuda:0}"
QWEN_BS="${QWEN_BS:-48}"
FIRERED_BS="${FIRERED_BS:-24}"
PIPELINE_PID="$ROOT/outputs/pipeline_L90pct.pid"
PIPELINE_LOG="$ROOT/outputs/pipeline_L90pct.log"
FILTER_PID="$ROOT/outputs/tier_A_filter_rest90.pid"
FILTER_LOG="$ROOT/outputs/tier_A_filter_rest90.log"

prepare_manifest() {
  if [[ -f "$MANIFEST" ]]; then
    local n
    n=$(wc -l <"$MANIFEST")
    echo "[prepare] manifest exists: $MANIFEST ($n lines) — skip rebuild"
    return 0
  fi
  echo "[prepare] building complement (~90%) manifest -> $MANIFEST"
  cd "$ROOT"
  "$PYTHON" sample_L_manifest.py \
    --fraction 0.1 \
    --complement \
    --seed 42 \
    --out "$MANIFEST"
  echo "[prepare] done: $(wc -l <"$MANIFEST") lines"
}

count_shards() {
  local dir="$1"
  if compgen -G "$dir/shard_*.jsonl" >/dev/null; then
    wc -l "$dir"/shard_*.jsonl | tail -1
  else
    echo "0 total"
  fi
}

alive_pid() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local p
  p=$(cat "$f" 2>/dev/null || true)
  [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null
}

start_bg() {
  local name="$1"
  local out_dir="$2"
  local log_file="$3"
  local pid_file="$4"
  local cmd="$5"

  mkdir -p "$out_dir"
  if alive_pid "$pid_file"; then
    echo "[$name] already running pid=$(cat "$pid_file")"
    return 0
  fi
  : >>"$log_file"
  echo "[$name] starting -> $log_file"
  nohup bash -c "$cmd" >>"$log_file" 2>&1 &
  echo $! >"$pid_file"
  echo "[$name] pid=$(cat "$pid_file")"
}

QWEN_PREFETCH="${QWEN_PREFETCH:-4}"
FIRERED_PREFETCH="${FIRERED_PREFETCH:-4}"

run_qwen_cmd() {
  cat <<EOF
cd "$ROOT" && \\
$PYTHON run_qwen3_asr_compare.py \\
  --manifest "$MANIFEST" \\
  --model "$QWEN_MODEL" \\
  --device "$DEVICE" \\
  --dtype bfloat16 \\
  --batch_size "$QWEN_BS" \\
  --prefetch_workers "$QWEN_PREFETCH" \\
  --log_every 100 \\
  --out_dir "$QWEN_OUT"
EOF
}

run_firered_cmd() {
  cat <<EOF
cd "$ROOT" && \\
export PYTHONPATH="$FIRERED_ROOT:\${PYTHONPATH:-}" && \\
export PATH="$FIRERED_ROOT/fireredasr2s:\${PATH:-}" && \\
$PYTHON run_firered_asr_compare.py \\
  --manifest "$MANIFEST" \\
  --model_dir "$FIRERED_MODEL" \\
  --batch_size "$FIRERED_BS" \\
  --prefetch_workers "$FIRERED_PREFETCH" \\
  --log_every 100 \\
  --out_dir "$FIRERED_OUT"
EOF
}

run_filter_cmd() {
  cat <<EOF
cd "$ROOT" && \\
$PYTHON filter_tier_A.py \\
  --qwen_dir "$QWEN_OUT" \\
  --firered_dir "$FIRERED_OUT" \\
  --out "$TIER_A_OUT" \\
  --summary "$TIER_A_SUMMARY"
EOF
}

run_pipeline_cmd() {
  cat <<EOF
set -euo pipefail
set +o noclobber
echo "[pipeline] === prepare ==="
# manifest prepared by launcher before bg
echo "[pipeline] === qwen ==="
$(run_qwen_cmd)
echo "[pipeline] === firered ==="
$(run_firered_cmd)
echo "[pipeline] === filter Tier-A ==="
$(run_filter_cmd)
echo "[pipeline] ALL DONE"
date
EOF
}

show_status() {
  echo "=== REST90 status ==="
  echo "manifest: $MANIFEST"
  if [[ -f "$MANIFEST" ]]; then
    echo "  lines=$(wc -l <"$MANIFEST")"
  else
    echo "  (missing — run: bash launch_rest90.sh prepare)"
  fi
  echo "qwen shards:    $(count_shards "$QWEN_OUT")"
  if alive_pid "$QWEN_OUT/run.pid"; then echo "  qwen RUNNING pid=$(cat "$QWEN_OUT/run.pid")"; fi
  echo "firered shards: $(count_shards "$FIRERED_OUT")"
  if alive_pid "$FIRERED_OUT/run.pid"; then echo "  firered RUNNING pid=$(cat "$FIRERED_OUT/run.pid")"; fi
  if alive_pid "$PIPELINE_PID"; then echo "pipeline RUNNING pid=$(cat "$PIPELINE_PID")"; fi
  if alive_pid "$FILTER_PID"; then echo "filter RUNNING pid=$(cat "$FILTER_PID")"; fi
  if [[ -f "$TIER_A_OUT" ]]; then echo "tierA: $(wc -l <"$TIER_A_OUT") -> $TIER_A_OUT"; fi
  if [[ -f "$TIER_A_SUMMARY" ]]; then echo "summary: $TIER_A_SUMMARY"; cat "$TIER_A_SUMMARY"; fi
  echo "logs:"
  echo "  $PIPELINE_LOG"
  echo "  $QWEN_OUT/run.log"
  echo "  $FIRERED_OUT/run.log"
  echo "  $FILTER_LOG"
}

stop_jobs() {
  for f in "$PIPELINE_PID" "$QWEN_OUT/run.pid" "$FIRERED_OUT/run.pid" "$FILTER_PID"; do
    if alive_pid "$f"; then
      local p
      p=$(cat "$f")
      echo "[stop] kill $p ($f)"
      # also kill child python filter if launcher is the pid
      pkill -P "$p" 2>/dev/null || true
      kill "$p" 2>/dev/null || true
    fi
  done
  # belt-and-suspenders for orphaned filter_tier_A
  pkill -f "filter_tier_A.py --qwen_dir $QWEN_OUT" 2>/dev/null || true
  echo "[stop] done (safe to re-launch; shards keep progress)"
}

case "$MODE" in
  prepare)
    prepare_manifest
    ;;
  qwen)
    prepare_manifest
    start_bg "qwen90" "$QWEN_OUT" "$QWEN_OUT/run.log" "$QWEN_OUT/run.pid" "$(run_qwen_cmd)"
    ;;
  firered)
    prepare_manifest
    start_bg "firered90" "$FIRERED_OUT" "$FIRERED_OUT/run.log" "$FIRERED_OUT/run.pid" "$(run_firered_cmd)"
    ;;
  filter)
    mkdir -p "$ROOT/outputs"
    if alive_pid "$FILTER_PID"; then
      echo "[filter] already running pid=$(cat "$FILTER_PID")"
      echo "[filter] monitor: tail -f $FILTER_LOG"
      exit 0
    fi
    # also detect an already-running filter started outside this launcher
    if pgrep -f "filter_tier_A.py --qwen_dir $QWEN_OUT" >/dev/null 2>&1; then
      echo "[filter] filter_tier_A.py already running (external); not starting another"
      pgrep -af "filter_tier_A.py --qwen_dir $QWEN_OUT" || true
      exit 0
    fi
    start_bg "filter90" "$ROOT/outputs" "$FILTER_LOG" "$FILTER_PID" "$(run_filter_cmd)"
    echo "[filter] monitor: tail -f $FILTER_LOG"
    echo "[filter] status:  bash launch_rest90.sh status"
    ;;
  status)
    show_status
    ;;
  stop)
    stop_jobs
    ;;
  all)
    prepare_manifest
    mkdir -p "$QWEN_OUT" "$FIRERED_OUT" "$ROOT/outputs"
    if alive_pid "$PIPELINE_PID"; then
      echo "[pipeline] already running pid=$(cat "$PIPELINE_PID")"
      exit 0
    fi
    # also refuse if a solo model job is holding the GPU
    if alive_pid "$QWEN_OUT/run.pid" || alive_pid "$FIRERED_OUT/run.pid"; then
      echo "[pipeline] qwen/firered already running; use status/stop first"
      exit 1
    fi
    : >>"$PIPELINE_LOG"
    echo "[pipeline] starting rest90 qwen->firered->filter -> $PIPELINE_LOG"
    nohup bash -c "$(run_pipeline_cmd)" >>"$PIPELINE_LOG" 2>&1 &
    echo $! >"$PIPELINE_PID"
    echo $! >"$QWEN_OUT/run.pid"
    echo "[pipeline] pid=$(cat "$PIPELINE_PID")"
    echo "[pipeline] monitor: tail -f $PIPELINE_LOG"
    echo "[pipeline] resume:  re-run bash launch_rest90.sh all   # skips finished utts"
    ;;
  *)
    echo "Usage: $0 {all|prepare|qwen|firered|filter|status|stop}"
    exit 1
    ;;
esac
