"""构音障碍音色克隆 — 零样本克隆推理。

基于 Qwen3-TTS generate_voice_clone，支持两种模式：
- xvec: x_vector_only_mode=True，仅提取说话人向量（对构音障碍参考更友好）
- icl:  x_vector_only_mode=False，ICL 上下文克隆（需 ref_text）

用法:
    python -m ranAudio.voice_clone.zeroshot --spk 06 --texts "今天天气真好" "我们一起去看电影吧"
    python -m ranAudio.voice_clone.zeroshot --spk 06 --from-test 10
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ranAudio.voice_clone.config import (
    PILOT_SPK,
    QWEN_TTS_BASE,
    ZS_GEN_KWARGS,
    zs_out_dir,
)
from ranAudio.voice_clone.utils import read_jsonl, save_audio

logger = logging.getLogger(__name__)


class VoiceCloneInferencer:
    """Qwen3-TTS 零样本音色克隆推理器。"""

    def __init__(self, model_path: str | Path | None = None, device: str = "cuda:0"):
        self.model_path = str(model_path or QWEN_TTS_BASE)
        self.device = device
        self.tts = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        # 自动选择 attention 实现
        attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
            logger.warning("flash_attn 未安装，使用 sdpa attention")

        logger.info("加载模型: %s (attn=%s)", self.model_path, attn_impl)
        self.tts = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        logger.info("模型加载完成")

    def create_prompt(self, ref_audio: str, ref_text: str = None, x_vector_only: bool = True):
        """预提取说话人 prompt（可复用，避免重复编码参考音频）。"""
        assert self.tts is not None, "请先调用 load()"
        return self.tts.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

    def generate(
        self,
        text: str | list[str],
        ref_audio: str = None,
        ref_text: str = None,
        voice_clone_prompt=None,
        x_vector_only: bool = True,
        language: str = "Chinese",
        **kwargs,
    ):
        """生成语音，返回 (wavs, sample_rate)。"""
        assert self.tts is not None, "请先调用 load()"
        gen_kwargs = {**ZS_GEN_KWARGS, **kwargs}

        if voice_clone_prompt is not None:
            return self.tts.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **gen_kwargs,
            )
        else:
            return self.tts.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                **gen_kwargs,
            )


def run_zeroshot(
    spk: str = PILOT_SPK,
    texts: list[str] | None = None,
    from_test: int = 0,
    mode: str = "xvec",
    model_path: str | None = None,
    max_items: int = 20,
) -> list[dict]:
    """执行零样本克隆，返回生成结果列表。"""
    from ranAudio.voice_clone.config import ref_dir

    # 1. 加载参考音频信息
    ref_info_path = ref_dir(spk) / "ref_candidates.jsonl"
    ref_candidates = read_jsonl(ref_info_path)
    if not ref_candidates:
        raise ValueError(f"未找到参考音频信息: {ref_info_path}")

    primary_ref = ref_candidates[0]
    ref_wav = primary_ref["wav"]
    ref_text = primary_ref["text"]
    logger.info("参考音频: %s  '%s'", ref_wav, ref_text)

    # 2. 确定要合成的文本
    if texts is not None:
        syn_texts = texts
        syn_ids = [f"custom_{i}" for i in range(len(texts))]
    elif from_test > 0:
        test_path = Path(f"runs/voice_clone/S{spk}/test_texts.jsonl")
        test_items = read_jsonl(test_path)[:from_test]
        syn_texts = [it["text"] for it in test_items]
        syn_ids = [it["id"] for it in test_items]
    else:
        raise ValueError("请指定 --texts 或 --from-test")

    if max_items and len(syn_texts) > max_items:
        syn_texts = syn_texts[:max_items]
        syn_ids = syn_ids[:max_items]

    logger.info("待合成文本: %d 条", len(syn_texts))

    # 3. 加载模型
    xvec = mode == "xvec"
    inferencer = VoiceCloneInferencer(model_path=model_path)
    inferencer.load()

    # 4. 预提取说话人 prompt（避免重复编码参考音频）
    prompt = inferencer.create_prompt(ref_audio=ref_wav, ref_text=ref_text, x_vector_only=xvec)
    logger.info("说话人 prompt 提取完成 (x_vector_only=%s)", xvec)

    # 5. 批量生成
    out_dir = zs_out_dir(spk, mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (utt_id, text) in enumerate(zip(syn_ids, syn_texts)):
        t0 = time.time()
        wavs, sr = inferencer.generate(text=text, voice_clone_prompt=prompt)
        elapsed = time.time() - t0

        out_path = out_dir / f"{utt_id}.wav"
        save_audio(out_path, wavs[0], sr)

        results.append(
            {
                "id": utt_id,
                "text": text,
                "wav": str(out_path),
                "time": round(elapsed, 3),
            }
        )
        logger.info("[%d/%d] %s -> %s (%.2fs)", i + 1, len(syn_texts), utt_id, out_path.name, elapsed)

    # 6. 写结果索引
    from ranAudio.voice_clone.utils import write_jsonl

    write_jsonl(out_dir / "results.jsonl", results)
    logger.info("零样本克隆完成: %d 条 -> %s", len(results), out_dir)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构音障碍音色克隆 — 零样本推理")
    parser.add_argument("--spk", type=str, default=PILOT_SPK)
    parser.add_argument("--texts", nargs="+", help="直接指定合成文本")
    parser.add_argument("--from-test", type=int, default=0, help="从测试集取 N 条文本")
    parser.add_argument("--mode", choices=["xvec", "icl"], default="xvec", help="克隆模式")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=20)
    args = parser.parse_args()

    results = run_zeroshot(
        spk=args.spk,
        texts=args.texts,
        from_test=args.from_test,
        mode=args.mode,
        model_path=args.model_path,
        max_items=args.max_items,
    )
    print(f"\n生成完成: {len(results)} 条")


if __name__ == "__main__":
    main()
