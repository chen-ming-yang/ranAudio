#!/usr/bin/env bash
# Setup FireRedASR2-AED using ModelScope mirror. Reuses existing clone if present.
set -euo pipefail

ROOT="${FIRERED_ROOT:-/cmy/cmy/FireRedASR2S}"
AED_DIR="$ROOT/pretrained_models/FireRedASR2-AED"
PYTHON="${PYTHON:-/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python}"

echo "[setup] root=$ROOT python=$PYTHON"

if [[ ! -d "$ROOT/fireredasr2s" ]]; then
  echo "[setup] cloning via ghproxy mirror..."
  git clone --depth 1 https://ghproxy.com/https://github.com/FireRedTeam/FireRedASR2S.git "$ROOT" \
    || git clone --depth 1 https://gitclone.com/github.com/FireRedTeam/FireRedASR2S.git "$ROOT"
else
  echo "[setup] code already present, skip clone"
fi

# Lightweight deps only (do NOT reinstall torch)
"$PYTHON" -m pip install -q cn2an kaldiio kaldi_native_fbank sentencepiece soundfile textgrid 'peft>=0.13.2' modelscope

mkdir -p "$ROOT/pretrained_models"

if [[ -f "$AED_DIR/model.pth.tar" ]]; then
  echo "[setup] AED weights complete: $AED_DIR/model.pth.tar"
else
  echo "[setup] downloading AED via ModelScope (CN mirror) ..."
  "$PYTHON" - <<PY
from modelscope import snapshot_download
p = snapshot_download(
    "xukaituo/FireRedASR2-AED",
    local_dir="$AED_DIR",
)
print("done", p)
PY
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PYTHON" - <<'PY'
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
print("FireRedAsr2 import OK")
PY
ls -lah "$AED_DIR" | head -20
echo "[setup] done"
