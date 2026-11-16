#!/usr/bin/env python3
import transformers
print("version", transformers.__version__)
try:
    from transformers import Qwen3ASRForConditionalGeneration, AutoProcessor
    print("Qwen3ASRForConditionalGeneration OK")
except Exception as e:
    print("Qwen3ASRFail", type(e).__name__, e)

try:
    from transformers import AutoModelForSpeechSeq2Seq
    print("AutoModelForSpeechSeq2Seq OK")
except Exception as e:
    print("SpeechSeq2SeqFail", e)

# list attrs with Qwen3ASR
names = [n for n in dir(transformers) if "Qwen3" in n or "ASR" in n]
print("attrs", names[:40])
