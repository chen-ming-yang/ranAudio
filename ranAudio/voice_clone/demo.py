"""构音障碍音色克隆 — 推理 Demo CLI。

输入说话人 ID + 文本 → 输出克隆语音 wav。

用法:
    # 零样本克隆（xvec 模式）
    python -m ranAudio.voice_clone.demo --spk 06 --text "今天天气真好"

    # 零样本克隆（ICL 模式）
    python -m ranAudio.voice_clone.demo --spk 06 --text "你好" --mode icl

    # 微调模型推理
    python -m ranAudio.voice_clone.demo --spk 06 --text "你好" --mode finetuned

    # 批量文本
    python -m ranAudio.voice_clone.demo --spk 06 --text "文本1" "文本2" --out-dir /tmp/output

    # 从文件读取文本（每行一条）
    python -m ranAudio.voice_clone.demo --spk 06 --text-file texts.txt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ranAudio.voice_clone.config import (
    PILOT_SPK,
    QWEN_TTS_BASE,
    ft_ckpt_dir,
    ref_dir,
    zs_out_dir,
)
from ranAudio.voice_clone.utils import read_jsonl, save_audio

logger = logging.getLogger(__name__)


def run_demo(
    spk: str = PILOT_SPK,
    texts: list[str] | None = None,
    text_file: str | None = None,
    mode: str = "xvec",
    out_dir: str | None = None,
    model_path: str | None = None,
) -> list[dict]:
    """执行推理 demo。"""
    # 1. 获取文本列表
    if texts:
        syn_texts = texts
    elif text_file:
        syn_texts = [line.strip() for line in Path(text_file).read_text().splitlines() if line.strip()]
    else:
        raise ValueError("请指定 --text 或 --text-file")

    # 2. 确定输出目录
    if out_dir:
        odir = Path(out_dir)
    elif mode == "finetuned":
        odir = ft_ckpt_dir(spk) / "demo_output"
    else:
        odir = zs_out_dir(spk, mode) / "demo_output"
    odir.mkdir(parents=True, exist_ok=True)

    # 3. 加载模型
    if mode == "finetuned":
        # 使用微调后的模型
        ckpt_dir = ft_ckpt_dir(spk)
        ckpts = sorted(ckpt_dir.glob("checkpoint-epoch-*"))
        if not ckpts:
            raise FileNotFoundError(f"未找到微调 checkpoint: {ckpt_dir}")
        use_model_path = str(ckpts[-1])
        logger.info("使用微调模型: %s", use_model_path)
    else:
        use_model_path = model_path or str(QWEN_TTS_BASE)
        logger.info("使用基础模型: %s", use_model_path)

    from ranAudio.voice_clone.zeroshot import VoiceCloneInferencer

    inferencer = VoiceCloneInferencer(model_path=use_model_path)
    inferencer.load()

    # 4. 创建说话人 prompt
    if mode == "finetuned":
        # 微调模型用 generate_custom_voice
        pass  # 单独处理
    else:
        ref_candidates = read_jsonl(ref_dir(spk) / "ref_candidates.jsonl")
        primary_ref = ref_candidates[0]
        xvec = mode == "xvec"
        prompt = inferencer.create_prompt(
            ref_audio=primary_ref["wav"],
            ref_text=primary_ref["text"],
            x_vector_only=xvec,
        )
        logger.info("说话人 prompt 提取完成 (mode=%s)", mode)

    # 5. 生成
    results = []
    for i, text in enumerate(syn_texts):
        t0 = time.time()

        if mode == "finetuned":
            # 微调模型推理（模型内部以 speaker.lower() 查表）
            # 限制长度防止生成不终止（曾出现 655s 失控音频）
            speaker_name = f"s{spk}"
            wavs, sr = inferencer.tts.generate_custom_voice(
                text=text,
                speaker=speaker_name,
                max_new_tokens=512,
                do_sample=False,
            )
        else:
            wavs, sr = inferencer.generate(text=text, voice_clone_prompt=prompt)

        elapsed = time.time() - t0

        out_path = odir / f"demo_{i:03d}.wav"
        save_audio(out_path, wavs[0], sr)

        results.append({
            "index": i,
            "text": text,
            "wav": str(out_path),
            "time": round(elapsed, 3),
        })
        logger.info("[%d/%d] '%s' -> %s (%.2fs)", i + 1, len(syn_texts), text[:20], out_path.name, elapsed)

    # 6. 写索引
    from ranAudio.voice_clone.utils import write_jsonl

    write_jsonl(odir / "demo_results.jsonl", results)

    logger.info("Demo 完成: %d 条 -> %s", len(results), odir)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构音障碍音色克隆 — 推理 Demo")
    parser.add_argument("--spk", type=str, default=PILOT_SPK, help="说话人 ID")
    parser.add_argument("--text", nargs="+", help="合成文本（可多条）")
    parser.add_argument("--text-file", type=str, help="文本文件路径（每行一条）")
    parser.add_argument("--mode", choices=["xvec", "icl", "finetuned"], default="xvec")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    results = run_demo(
        spk=args.spk,
        texts=args.text,
        text_file=args.text_file,
        mode=args.mode,
        out_dir=args.out_dir,
        model_path=args.model_path,
    )
    print(f"\n生成完成: {len(results)} 条")
    for r in results:
        print(f"  {r['wav']}")


if __name__ == "__main__":
    main()
