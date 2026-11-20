#!/usr/bin/env bash
# Wait until GPU has enough free memory, then train S06 char-align.
set -euo pipefail
source /cmy/cmy/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-asr
cd /cmy/cmy/ranAudio
export PYTHONPATH="/cmy/cmy/real-asr:${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

NEED_MIB="${NEED_MIB:-8000}"
POLL_SEC="${POLL_SEC:-60}"
OUT_DIR="${OUT_DIR:-runs/char_align/s06_part1}"
LOG="${LOG:-tmp/char_align_train.log}"

echo "[$(date)] waiting for >= ${NEED_MIB} MiB free on GPU0 ..." | tee -a "$LOG"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[$(date)] free_mib=${free}" | tee -a "$LOG"
  if [[ "${free}" -ge "${NEED_MIB}" ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[$(date)] starting train → ${OUT_DIR}" | tee -a "$LOG"
python char_align/train_align.py \
  --device cuda \
  --epochs 15 \
  --batch-size 8 \
  --num-workers 2 \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] eval test" | tee -a "$LOG"
python char_align/eval_align.py \
  --split test \
  --baseline whisperx \
  --checkpoint "${OUT_DIR}/best.pt" \
  --out-json tmp/char_align_test_final.json \
  2>&1 | tee -a "$LOG"
