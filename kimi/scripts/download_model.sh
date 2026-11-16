#!/usr/bin/env bash
# Download / resume Kimi-Audio-7B-Instruct via ModelScope (国内镜像).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/models/Kimi-Audio-7B-Instruct}"
PY="${KIMI_PY:-/cmy/cmy/miniconda3/envs/kimi-audio/bin/python}"
mkdir -p "$OUT"
exec "$PY" -u - <<PY
from modelscope.hub.snapshot_download import snapshot_download
print("download -> $OUT", flush=True)
path = snapshot_download(
    "moonshotai/Kimi-Audio-7B-Instruct",
    local_dir="$OUT",
)
print("DONE", path, flush=True)
PY
