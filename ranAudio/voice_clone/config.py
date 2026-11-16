"""构音障碍音色克隆 — 配置常量。"""

from __future__ import annotations

from pathlib import Path

from ranAudio.paths import DATA_DIR, QWEN_BASE, QWEN3_TTS_ROOT, RANAUDIO, RUNS_DIR

# ── 数据 ──────────────────────────────────────────────
DYSARTHRIC_JSONL = DATA_DIR / "dysarthric.jsonl"
AUDIO_ROOT = Path("/cmy/data/after_catting/10h/Audio")

# 试点说话人
PILOT_SPK = "06"

# ── Qwen3-TTS 模型 ────────────────────────────────────
QWEN_TTS_BASE = QWEN_BASE                       # 0.6B-Base（voice clone + 微调初始化）
QWEN_TTS_TOKENIZER = QWEN3_TTS_ROOT / "Qwen3-TTS-Tokenizer-12Hz"
QWEN_FT_DIR = QWEN3_TTS_ROOT / "finetuning"

# ── 输出 ──────────────────────────────────────────────
VC_RUNS = RUNS_DIR / "voice_clone"

# ── 零样本克隆 ────────────────────────────────────────
ZS_GEN_KWARGS = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)

# ── 微调 ──────────────────────────────────────────────
FT_BATCH_SIZE = 2
FT_LR = 2e-6
FT_EPOCHS = 5
FT_TRAIN_SAMPLES = 3000        # 每位说话人取训练样本数
FT_VAL_SAMPLES = 200           # 验证集样本数

# ── 评估 ──────────────────────────────────────────────
# ECAPA-TDNN（speechbrain）
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# ── 参考音频筛选 ──────────────────────────────────────
REF_MIN_DURATION = 3.0         # 参考音频最短时长（秒）
REF_CANDIDATES = 5             # 候选参考音频数量


def spk_dir(spk: str) -> Path:
    """返回指定说话人的输出目录。"""
    return VC_RUNS / f"S{spk}"


def ref_dir(spk: str) -> Path:
    return spk_dir(spk) / "ref"


def zs_out_dir(spk: str, mode: str = "xvec") -> Path:
    """mode: 'xvec' 或 'icl'"""
    return spk_dir(spk) / f"zeroshot_{mode}"


def ft_out_dir(spk: str) -> Path:
    return spk_dir(spk) / "finetuned"


def ft_ckpt_dir(spk: str) -> Path:
    return spk_dir(spk) / "checkpoint"
