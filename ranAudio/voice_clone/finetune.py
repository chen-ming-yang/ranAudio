"""构音障碍音色克隆 — 微调训练。

整合 Qwen3-TTS finetuning 工具链：
1. prepare_data.py: 提取 audio_codes
2. sft_12hz.py: 单说话人 SFT

用法:
    python -m ranAudio.voice_clone.finetune --spk 06
    python -m ranAudio.voice_clone.finetune --spk 06 --epochs 3 --lr 2e-6
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ranAudio.voice_clone.config import (
    FT_BATCH_SIZE,
    FT_EPOCHS,
    FT_LR,
    PILOT_SPK,
    QWEN_FT_DIR,
    QWEN_TTS_BASE,
    QWEN_TTS_TOKENIZER,
    ft_ckpt_dir,
    spk_dir,
)

logger = logging.getLogger(__name__)

QWEN_TTS_ENV_PY = "/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python"


def prepare_finetune_data(
    train_jsonl: Path,
    output_jsonl: Path,
    device: str = "cuda:0",
) -> Path:
    """调用 Qwen3-TTS prepare_data.py 提取 audio_codes。"""
    if output_jsonl.exists():
        logger.info("微调数据已存在，跳过: %s", output_jsonl)
        return output_jsonl

    cmd = [
        QWEN_TTS_ENV_PY,
        str(QWEN_FT_DIR / "prepare_data.py"),
        "--device", device,
        "--tokenizer_model_path", str(QWEN_TTS_TOKENIZER),
        "--input_jsonl", str(train_jsonl),
        "--output_jsonl", str(output_jsonl),
    ]
    logger.info("提取 audio_codes: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(QWEN_FT_DIR))
    if result.returncode != 0:
        logger.error("prepare_data 失败:\n%s", result.stderr)
        raise RuntimeError(f"prepare_data.py failed: {result.stderr[-500:]}")

    logger.info("audio_codes 提取完成: %s", output_jsonl)
    return output_jsonl


def run_sft(
    train_jsonl_with_codes: Path,
    output_dir: Path,
    speaker_name: str,
    batch_size: int = FT_BATCH_SIZE,
    lr: float = FT_LR,
    num_epochs: int = FT_EPOCHS,
    init_model_path: str | Path | None = None,
) -> Path:
    """调用 sft_12hz.py 执行单说话人微调。"""
    init_model = str(init_model_path or QWEN_TTS_BASE)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用包装脚本（自动降级 flash_attention_2 → sdpa）
    sft_script = QWEN_FT_DIR / "sft_wrapper.py"
    if not sft_script.exists():
        sft_script = QWEN_FT_DIR / "sft_12hz.py"

    cmd = [
        QWEN_TTS_ENV_PY,
        str(sft_script),
        "--init_model_path", init_model,
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_jsonl_with_codes),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(num_epochs),
        "--speaker_name", speaker_name,
    ]
    logger.info("开始微调: %s", " ".join(cmd))
    logger.info("输出目录: %s", output_dir)

    result = subprocess.run(cmd, cwd=str(QWEN_FT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"sft_12hz.py failed (rc={result.returncode})")

    # 找到最后一个 checkpoint
    ckpts = sorted(output_dir.glob("checkpoint-epoch-*"))
    if ckpts:
        logger.info("微调完成，最后 checkpoint: %s", ckpts[-1])
        return ckpts[-1]
    else:
        raise RuntimeError(f"未找到 checkpoint: {output_dir}")


def run_finetune(
    spk: str = PILOT_SPK,
    batch_size: int = FT_BATCH_SIZE,
    lr: float = FT_LR,
    epochs: int = FT_EPOCHS,
    device: str = "cuda:0",
) -> dict:
    """执行完整微调流程。"""
    sdir = spk_dir(spk)

    # 1. 检查训练数据
    train_jsonl = sdir / "finetune_train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_jsonl}，请先运行 data_prep")

    # 2. 提取 audio_codes
    train_codes_jsonl = sdir / "finetune_train_with_codes.jsonl"
    prepare_finetune_data(train_jsonl, train_codes_jsonl, device=device)

    # 3. 执行 SFT（模型内部以 speaker.lower() 查表，必须用小写）
    speaker_name = f"s{spk}"
    ckpt_dir = ft_ckpt_dir(spk)
    best_ckpt = run_sft(
        train_jsonl_with_codes=train_codes_jsonl,
        output_dir=ckpt_dir,
        speaker_name=speaker_name,
        batch_size=batch_size,
        lr=lr,
        num_epochs=epochs,
    )

    result = {
        "spk": spk,
        "speaker_name": speaker_name,
        "train_jsonl": str(train_jsonl),
        "train_codes_jsonl": str(train_codes_jsonl),
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoint": str(best_ckpt),
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
    }
    logger.info("微调结果: %s", result)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构音障碍音色克隆 — 微调训练")
    parser.add_argument("--spk", type=str, default=PILOT_SPK)
    parser.add_argument("--batch-size", type=int, default=FT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FT_LR)
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    result = run_finetune(
        spk=args.spk,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
    print("\n=== 微调完成 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
