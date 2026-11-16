import os, time, torch
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from kimia_infer.api.kimia import KimiAudio
MP="/cmy/cmy/ranAudio/kimi/models/Kimi-Audio-7B-Instruct"
t0=time.time()
model=KimiAudio(model_path=MP, load_detokenizer=False)   # 纯 ASR,不需要声码器
print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)
f,t=torch.cuda.mem_get_info()
print(f"[probe] after load: free={f/2**30:.1f}GiB used={(t-f)/2**30:.1f}GiB", flush=True)
sp={"audio_temperature":0.8,"audio_top_k":10,"text_temperature":0.0,"text_top_k":5,
    "audio_repetition_penalty":1.0,"audio_repetition_window_size":64,
    "text_repetition_penalty":1.0,"text_repetition_window_size":16}
msgs=[{"role":"user","message_type":"text","content":"Please transcribe the following audio:"},
      {"role":"user","message_type":"audio","content":"/cmy/cmy/ranAudio/kimi/test_audios/asr_example.wav"}]
t1=time.time()
_, txt = model.generate(msgs, **sp, output_type="text")
print(f"[probe] ASR took {time.time()-t1:.1f}s", flush=True)
print("[probe] ASR OUTPUT:", txt, flush=True)
f2,t2=torch.cuda.mem_get_info()
print(f"[probe] after infer: free={f2/2**30:.1f}GiB", flush=True)
