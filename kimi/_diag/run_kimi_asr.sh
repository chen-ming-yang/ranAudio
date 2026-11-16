#!/usr/bin/env bash
# Kimi-Audio 在本机的运行封装(已实测通过)
# 用法: bash run_kimi_asr.sh <wav路径> [输出文件]
set -euo pipefail
cd /cmy/cmy/ranAudio/kimi
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
PY=/cmy/cmy/miniconda3/envs/kimi-audio/bin/python
WAV=${1:?用法: run_kimi_asr.sh <wav> [out.txt]}
OUT=${2:-/tmp/kimi_asr_out.txt}

"$PY" -u - "$WAV" "$OUT" <<"PY"
import sys, torch
from kimia_infer.api.kimia import KimiAudio
wav, outp = sys.argv[1], sys.argv[2]
MP="/cmy/cmy/ranAudio/kimi/models/Kimi-Audio-7B-Instruct"
model=KimiAudio(model_path=MP, load_detokenizer=False)  # 纯 ASR;需要出音频时改 True
sp={"audio_temperature":0.8,"audio_top_k":10,"text_temperature":0.0,"text_top_k":5,
    "audio_repetition_penalty":1.0,"audio_repetition_window_size":64,
    "text_repetition_penalty":1.0,"text_repetition_window_size":16}
msgs=[{"role":"user","message_type":"text","content":"Please transcribe the following audio:"},
      {"role":"user","message_type":"audio","content":wav}]
_, txt = model.generate(msgs, **sp, output_type="text")
print(txt)
open(outp,"w").write(txt)
f,t=torch.cuda.mem_get_info()
print(f"[mem] free={f/2**30:.1f}GiB / total={t/2**30:.1f}GiB", file=sys.stderr)
PY
