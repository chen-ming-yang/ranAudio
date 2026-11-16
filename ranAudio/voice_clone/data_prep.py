"""构音障碍音色克隆 — 数据准备。

解析 MDSD 构音障碍数据集（dysarthric.jsonl），按说话人分组：
- 筛选参考音频（较长且清晰的音频）
- 划分 train / val / test
- 生成微调 JSONL（audio / text / ref_audio 格式）

用法:
    python -m ranAudio.voice_clone.data_prep --spk 06
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ranAudio.voice_clone.config import (
    AUDIO_ROOT,
    DYSARTHRIC_JSONL,
    FT_TRAIN_SAMPLES,
    FT_VAL_SAMPLES,
    PILOT_SPK,
    REF_CANDIDATES,
    REF_MIN_DURATION,
    ft_out_dir,
    ref_dir,
    spk_dir,
)
from ranAudio.voice_clone.utils import read_jsonl, write_jsonl

import logging

logger = logging.getLogger(__name__)


def load_speaker_data(spk: str) -> list[dict]:
    """加载指定说话人的所有数据条目。"""
    spk_tag = f"/Audio/{spk}/"
    items = read_jsonl(DYSARTHRIC_JSONL)
    spk_items = [it for it in items if spk_tag in it["wav"]]
    logger.info("说话人 S%s: 共 %d 条", spk, len(spk_items))
    return spk_items


def select_ref_audio(items: list[dict], n_candidates: int = REF_CANDIDATES) -> list[dict]:
    """筛选参考音频候选：按时长降序，取前 n 条。"""
    sorted_items = sorted(items, key=lambda x: x["duration"], reverse=True)
    candidates = [it for it in sorted_items if it["duration"] >= REF_MIN_DURATION]
    selected = candidates[:n_candidates]
    logger.info("参考音频候选（>%.1fs）: %d 条，取前 %d 条", REF_MIN_DURATION, len(candidates), n_candidates)
    for it in selected:
        logger.info("  %s: %.2fs  '%s'", it["id"], it["duration"], it["text"])
    return selected


def split_dataset(
    items: list[dict],
    train_n: int = FT_TRAIN_SAMPLES,
    val_n: int = FT_VAL_SAMPLES,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """划分 train / val / test。"""
    # 排除参考音频（避免泄漏）
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    # 过滤太短的音频（<0.5s 对微调意义不大）
    usable = [it for it in shuffled if it["duration"] >= 0.5]

    train = usable[:train_n]
    val = usable[train_n : train_n + val_n]
    test = usable[train_n + val_n : train_n + val_n + 200]
    logger.info("数据划分: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def build_finetune_jsonl(
    items: list[dict],
    ref_wav: str,
    out_path: Path,
) -> None:
    """生成 Qwen3-TTS 微调格式 JSONL。"""
    records = []
    for it in items:
        records.append(
            {
                "audio": it["wav"],
                "text": it["text"],
                "ref_audio": ref_wav,
            }
        )
    write_jsonl(out_path, records)
    logger.info("微调 JSONL: %s (%d 条)", out_path, len(records))


def run_data_prep(spk: str = PILOT_SPK) -> dict:
    """执行完整数据准备流程，返回产物路径。"""
    # 1. 加载数据
    items = load_speaker_data(spk)
    if not items:
        raise ValueError(f"说话人 S{spk} 无数据")

    # 2. 选择参考音频
    ref_candidates = select_ref_audio(items)
    if not ref_candidates:
        raise ValueError(f"说话人 S{spk} 无满足条件的参考音频（>{REF_MIN_DURATION}s）")

    # 参考音频目录
    rdir = ref_dir(spk)
    rdir.mkdir(parents=True, exist_ok=True)

    # 写参考音频信息
    ref_info_path = rdir / "ref_candidates.jsonl"
    write_jsonl(ref_info_path, ref_candidates)

    # 主参考音频 = 最长的那条
    primary_ref = ref_candidates[0]
    primary_ref_wav = primary_ref["wav"]
    logger.info("主参考音频: %s (%.2fs) '%s'", primary_ref["id"], primary_ref["duration"], primary_ref["text"])

    # 3. 划分数据集（排除参考音频）
    ref_ids = {it["id"] for it in ref_candidates}
    non_ref_items = [it for it in items if it["id"] not in ref_ids]
    train, val, test = split_dataset(non_ref_items)

    # 4. 生成微调 JSONL
    sdir = spk_dir(spk)
    sdir.mkdir(parents=True, exist_ok=True)

    build_finetune_jsonl(train, primary_ref_wav, sdir / "finetune_train.jsonl")
    build_finetune_jsonl(val, primary_ref_wav, sdir / "finetune_val.jsonl")
    build_finetune_jsonl(test, primary_ref_wav, sdir / "finetune_test.jsonl")

    # 5. 保存测试文本（用于零样本克隆测试）
    test_texts = [{"id": it["id"], "text": it["text"], "wav": it["wav"], "duration": it["duration"]} for it in test]
    write_jsonl(sdir / "test_texts.jsonl", test_texts)

    result = {
        "spk": spk,
        "total": len(items),
        "ref_primary": primary_ref["wav"],
        "ref_text": primary_ref["text"],
        "ref_candidates": str(ref_info_path),
        "train_jsonl": str(sdir / "finetune_train.jsonl"),
        "val_jsonl": str(sdir / "finetune_val.jsonl"),
        "test_jsonl": str(sdir / "finetune_test.jsonl"),
        "test_texts": str(sdir / "test_texts.jsonl"),
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
    }
    logger.info("数据准备完成: %s", result)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构音障碍音色克隆 — 数据准备")
    parser.add_argument("--spk", type=str, default=PILOT_SPK, help="说话人 ID（默认 06）")
    args = parser.parse_args()

    result = run_data_prep(args.spk)
    print("\n=== 数据准备结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
