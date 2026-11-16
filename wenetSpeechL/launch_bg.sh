#!/usr/bin/env bash
# Background launcher for Qwen3 / FireRed L-10% compare jobs (serial on one GPU).
# Usage:
#   bash launch_bg.sh all      # qwen then firered (wait for qwen finish)
#   bash launch_bg.sh qwen
#   bash launch_bg.sh firered
set -euo pipefail

ROOT="/cmy/cmy/ranAudio/wenetSpeechL"
MANIFEST="${MANIFEST:-$ROOT/sample_manifest.jsonl}"
MODE="${1:-all}"
PYTHON="${PYTHON:-/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python}"

QWEN_OUT="${QWEN_OUT:-$ROOT/outputs/qwen3_asr_L10pct}"
FIRERED_OUT="${FIRERED_OUT:-$ROOT/outputs/firered_asr_L10pct}"
QWEN_MODEL="${QWEN_MODEL:-/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B}"
FIRERED_MODEL="${FIRERED_MODEL:-/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED}"
FIRERED_ROOT="${FIRERED_ROOT:-/cmy/cmy/FireRedASR2S}"

start_one() {
  local name="$1"
  local out_dir="$2"
  local cmd="$3"
  mkdir -p "$out_dir"
  local pid_file="$out_dir/run.pid"
  local log_file="$out_dir/run.log"

  if [[ -f "$pid_file" ]]; then
    local old
    old="$(cat "$pid_file" || true)"
    if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
      echo "[$name] already running pid=$old"
      return 0
    fi
  fi

  echo "[$name] starting -> $log_file"
  nohup env -u PYTHONPATH bash -c "$cmd" >>"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "[$name] pid=$pid"
}

run_qwen_cmd() {
  cat <<EOF
cd "$ROOT" && \\
$PYTHON run_qwen3_asr_compare.py \\
  --manifest "$MANIFEST" \\
  --model "$QWEN_MODEL" \\
  --device cuda:0 \\
  --dtype bfloat16 \\
  --batch_size "${QWEN_BS:-48}" \\
  --prefetch_workers "${QWEN_PREFETCH:-4}" \\
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
  --batch_size "${FIRERED_BS:-24}" \\
  --prefetch_workers "${FIRERED_PREFETCH:-4}" \\
  --out_dir "$FIRERED_OUT"
EOF
}

run_all_serial_cmd() {
  # Wait until qwen finishes, then start firered in same nohup process chain
  cat <<EOF
set -e
$(run_qwen_cmd)
echo "[launcher] qwen finished, starting firered..."
$(run_firered_cmd)
echo "[launcher] all done"
EOF
}

case "$MODE" in
  qwen)
    start_one "qwen" "$QWEN_OUT" "$(run_qwen_cmd)"
    ;;
  firered)
    start_one "firered" "$FIRERED_OUT" "$(run_firered_cmd)"
    ;;
  all)
    # Single background process: qwen then firered (one GPU)
    mkdir -p "$QWEN_OUT" "$FIRERED_OUT"
    pid_file="$ROOT/outputs/pipeline_L10pct.pid"
    log_file="$ROOT/outputs/pipeline_L10pct.log"
    if [[ -f "$pid_file" ]]; then
      old="$(cat "$pid_file" || true)"
      if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
        echo "[pipeline] already running pid=$old"
        exit 0
      fi
    fi
    echo "[pipeline] starting serial qwen->firered -> $log_file"
    nohup bash -c "$(run_all_serial_cmd)" >>"$log_file" 2>&1 &
    echo $! >"$pid_file"
    # also mirror pids for convenience
    echo $! >"$QWEN_OUT/run.pid"
    echo "[pipeline] pid=$(cat "$pid_file")"
    ;;
  *)
    echo "Usage: $0 {all|qwen|firered}"
    exit 1
    ;;
esac
