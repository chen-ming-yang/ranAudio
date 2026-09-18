"""MiniCPM5-2B + Whisper encoder: turn-based car speech.

Project lives at /cmy/cmy/ranAudio (sibling of MiniCPM-o-Demo).
v1.0: speech -> MiniCPM5 XML FC.
v1.1: speech -> judge JSON {asr, need_fc, reply?} -> optional Qwen Slave.
S1.5/S2: WenetSpeech ASR (2B frozen) then <mode:speak|route> LoRA.

Not MiniCPM-o duplex. Does not touch py_backend or Gateway.
"""

from ranAudio.paths import bootstrap_sys_path

bootstrap_sys_path()
