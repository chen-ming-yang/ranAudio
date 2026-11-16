#!/usr/bin/env bash
# Kimi-Audio local experience page. Default text-only (no detokenizer).
# HTTP on 7861 (localhost mic OK) + HTTPS on 7862 (LAN IP mic). Log: kimi/demo.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT" "$ROOT/certs"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# HuggingFace hub often blocked; prefer mirror + local glm tokenizer
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export KIMI_GLM_TOKENIZER="${KIMI_GLM_TOKENIZER:-$ROOT/models/glm-4-voice-tokenizer}"
export KIMI_SENSEVOICE="${KIMI_SENSEVOICE:-/cmy/cmy/MiniCPM-o-Demo/models/iic/SenseVoiceSmall}"
PY="${KIMI_PY:-/cmy/cmy/miniconda3/envs/kimi-audio/bin/python}"
PORT="${KIMI_DEMO_PORT:-7861}"
CERT="$ROOT/certs/cert.pem"
KEY="$ROOT/certs/key.pem"

if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
  openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=kimi-audio-demo" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
  echo "wrote self-signed TLS $CERT"
fi

EXTRA=(--https --cert "$CERT" --key "$KEY")
if [[ "${KIMI_DEMO_HTTPS_ONLY:-0}" == "1" ]]; then
  EXTRA=(--https-only --cert "$CERT" --key "$KEY")
elif [[ "${KIMI_DEMO_HTTP_ONLY:-0}" == "1" ]]; then
  EXTRA=()
fi
if [[ "${KIMI_DEMO_SPEECH:-0}" == "1" ]]; then
  EXTRA+=(--speech)
fi
# 书面化：转写后默认走 LLM 语句规范化（也可用页面勾选 / ?spoken2written=1）
if [[ "${KIMI_SPOKEN2WRITTEN:-0}" == "1" ]]; then
  EXTRA+=(--spoken2written)
fi
echo $$ > "$ROOT/demo.pid"
exec "$PY" -u scripts/demo_web.py --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}"
