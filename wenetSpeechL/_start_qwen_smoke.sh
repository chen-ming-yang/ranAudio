#!/usr/bin/env bash
set +o noclobber
set -euo pipefail
cd /cmy/cmy/ranAudio/wenetSpeechL
OUT=outputs/qwen3_asr_L10pct_smoke
mkdir -p "$OUT"
rm -f "$OUT"/shard_*.jsonl
: > "$OUT/run.log"
head -n 200 sample_manifest.jsonl > sample_manifest_smoke200.jsonl
PY=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python
nohup env -u PYTHONPATH "$PY" run_qwen3_asr_compare.py \
  --manifest sample_manifest_smoke200.jsonl \
  --model /cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 8 \
  --max_samples 200 \
  --out_dir "$OUT" \
  --log_every 20 \
  >> "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "started pid=$(cat $OUT/run.pid)"
