"""构音障碍音色克隆 — 音频重采样工具。

将 MDSD 16kHz 音频重采样到 24kHz（Qwen3-TTS 要求）。

用法:
    python -m ranAudio.voice_clone.resample --spk 06
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ranAudio.voice_clone.config import PILOT_SPK, spk_dir
from ranAudio.voice_clone.utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

TARGET_SR = 24000


def resample_dataset(spk: str = PILOT_SPK) -> Path:
    """重采样微调数据集到 24kHz。"""
    import librosa
    import soundfile as sf

    sdir = spk_dir(spk)
    train_jsonl = sdir / "finetune_train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(f"请先运行 data_prep: {train_jsonl}")

    items = read_jsonl(train_jsonl)
    out_dir = sdir / "audio_24k"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_items = []
    for i, item in enumerate(items):
        src_wav = item["audio"]
        dst_wav = out_dir / Path(src_wav).name

        if not dst_wav.exists():
            wav, sr = sf.read(src_wav, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(str(dst_wav), wav, TARGET_SR)

        new_items.append({
            "audio": str(dst_wav),
            "text": item["text"],
            "ref_audio": item["ref_audio"],
        })

        if (i + 1) % 500 == 0:
            logger.info("重采样进度: %d/%d", i + 1, len(items))

    # 重采样参考音频
    ref_wav_src = items[0]["ref_audio"]
    ref_wav_dst = out_dir / Path(ref_wav_src).name
    if not ref_wav_dst.exists():
        wav, sr = sf.read(ref_wav_src, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(str(ref_wav_dst), wav, TARGET_SR)

    # 更新 ref_audio 路径
    for item in new_items:
        item["ref_audio"] = str(ref_wav_dst)

    out_jsonl = sdir / "finetune_train_24k.jsonl"
    write_jsonl(out_jsonl, new_items)
    logger.info("重采样完成: %d 条 -> %s", len(new_items), out_jsonl)
    return out_jsonl


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="音频重采样到 24kHz")
    parser.add_argument("--spk", type=str, default=PILOT_SPK)
    args = parser.parse_args()

    out = resample_dataset(args.spk)
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
