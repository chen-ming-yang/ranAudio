#!/usr/bin/env python3
"""为 ModelScope 批量制作发布目录（只挑可发布文件，剔除 optimizer 等训练中间态）。

用法:
    python3 build_publish_staging.py            # 生成
    python3 build_publish_staging.py --check    # 只检查源是否存在，不创建

特性:
  * 优先 os.link 硬链接（同文件系统，零额外占用、瞬时完成）
  * 跨设备时自动回退为复制
  * 生成 MANIFEST.tsv 供人工核验
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("/cmy/publish_ms")
BASE = Path("/cmy/cmy")

# 训练中间态：绝不发布
TRAIN_JUNK = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "scaler.pt",
}
TRAIN_JUNK_DIRS = {"__pycache__", ".git", "logs", "stale_base_shards"}

# ---------------------------------------------------------------- 仓库定义
# 每个 spec: (ModelScope 仓库名, [(源路径, 仓库内相对路径)], 额外排除的 glob)
SPECS: list[tuple[str, list[tuple[str, str]], set[str]]] = [
    # 1) Qwen3-ASR 微调：9 个实验各取最终 checkpoint
    (
        "qwen3-asr-ft-wenetspeech",
        [
            ("Qwen3-ASR/qwen3-asr-finetune-again-out-20260512", "finetune-again-20260512"),
            ("Qwen3-ASR/qwen3-asr-finetuning-out-bak", "finetuning-out-bak"),
            ("Qwen3-ASR/finetuning/best_model", "finetuning-best"),
            ("Qwen3-ASR/qwen3-asr-finetuning-out-20260615", "finetuning-out-20260615"),
            ("Qwen3-ASR/qwen3-asr-finetuning-out-20260524", "finetuning-out-20260524"),
            ("Qwen3-ASR/qwen3-asr-finetuning-out-20260509", "finetuning-out-20260509"),
            ("Qwen3-ASR/qwen3-asr-finetuning-all-out-20260616", "finetuning-all-out-20260616"),
            ("Qwen3-ASR/qwen3-asr-finetuning-all-out-20260524", "finetuning-all-out-20260524"),
            ("Qwen3-ASR/outputs/qwen3-asr-gspo-out", "gspo-final"),
            ("Qwen3-ASR/LICENSE", "LICENSE"),
            ("Qwen3-ASR/NOTICE", "NOTICE"),
        ],
        set(),
    ),
    # 2) Qwen3.5-0.8B merged（排除 gguf 量化件与 stale 残留）
    (
        "qwen3.5-0.8b-merged",
        [
            ("Qwen/Qwen3.5-0.8B-merged", "merged"),
            ("Qwen/Qwen3.5-0.8B-merged-v4", "merged-v4"),
            ("Qwen/Qwen3.5-0.8B-merged/LICENSE", "LICENSE"),
        ],
        {"*.gguf", "stale_base_shards/*"},
    ),
    # 3) ranAudio S2 LoRA（6 个 adapter）
    (
        "ranAudio-s2-lora",
        [
            ("ranAudio/runs/s2_fc_nofc/chat/adapter_final", "chat/adapter_final"),
            ("ranAudio/runs/s2_fc_nofc/chat/adapter_epoch1", "chat/adapter_epoch1"),
            ("ranAudio/runs/s2_fc_nofc/asr/adapter_final", "asr/adapter_final"),
            ("ranAudio/runs/s2_fc_nofc/asr/adapter_epoch1", "asr/adapter_epoch1"),
            ("ranAudio/runs/s2_fc_nofc/route/adapter_final", "route/adapter_final"),
            ("ranAudio/runs/s2_fc_nofc/route/adapter_epoch1", "route/adapter_epoch1"),
            ("ranAudio/runs/s2_fc_nofc/chat/projector_final.pt", "chat/projector_final.pt"),
            ("ranAudio/runs/s2_fc_nofc/asr/projector_final.pt", "asr/projector_final.pt"),
            ("ranAudio/runs/s2_fc_nofc/route/projector_final.pt", "route/projector_final.pt"),
            ("ranAudio/LICENSE", "LICENSE"),
            ("ranAudio/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        ],
        set(),
    ),
    # 4) cdsd MoE LoRA（10 个 run 的 final ckpt + dense_lora）
    ("cdsd-moe-lora", [], set()),
    # 5) MeanSE MeanFlow
    (
        "MeanSE-meanflow",
        [
            ("MeanSE/exp/meanflow/ncsnpp", "ncsnpp"),
            ("MeanSE/exp/meanflow/ncsnpp_dysar", "ncsnpp_dysar"),
            ("MeanSE/LICENSE", "LICENSE"),
            ("MeanSE/NOTICE", "NOTICE"),
        ],
        set(),
    ),
    # 7) ranAudio S1.5 ASR（各 run 的 best 权重）
    ("ranAudio-s15-asr", [], set()),
    # 8) ranAudio S2 full（7 分支 final）
    ("ranAudio-s2-full", [], set()),
]

S15_RUNS = ["a", "b_s", "b_ms", "b_ms_llm", "b_ms_cont", "dysarthric", "dysarthric_e2", "smoke"]
S2_FULL_BRANCHES = ["asr", "chat", "route", "mix", "mix_proj", "mix_llmcont", "mix_boundary"]


def expand_dynamic() -> None:
    """把需要在运行时枚举的源补进 SPECS。"""
    global SPECS
    cdsd_src: list[tuple[str, str]] = []
    base = Path("/cmy/cmy/ranAudio/cdsd/results")
    for spk in sorted(base.glob("moe_spk*")):
        cks = sorted(
            [p for p in spk.glob("moe*_lam*/checkpoint-*") if p.is_dir()],
            key=lambda p: (p.parent.name, int(p.name.rsplit("-", 1)[1])),
        )
        latest: dict[str, Path] = {}
        for c in cks:
            latest[c.parent.name] = c  # 按名字覆盖 → 留最大 step
        for run, path in latest.items():
            cdsd_src.append((str(path), f"{spk.name}/{run}"))
        lora = sorted(
            [p for p in spk.glob("dense_lora*/ckpt/checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.rsplit("-", 1)[1]),
        )
        if lora:
            cdsd_src.append((str(lora[-1]), f"{spk.name}/dense_lora"))
    cdsd_src.append(("/cmy/cmy/ranAudio/cdsd/LICENSE", "LICENSE"))
    cdsd_src.append(("/cmy/cmy/ranAudio/cdsd/NOTICE", "NOTICE"))

    s15_src: list[tuple[str, str]] = []
    for run in S15_RUNS:
        d = Path("/cmy/cmy/ranAudio/runs/s15_asr") / run
        for f in ("encoder_best.pt", "projector_best.pt", "args.json", "best.json"):
            p = d / f
            if p.is_file():
                s15_src.append((str(p), f"{run}/{f}"))
    s15_src.append(("/cmy/cmy/ranAudio/LICENSE", "LICENSE"))
    s15_src.append(("/cmy/cmy/ranAudio/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"))

    s2full_src: list[tuple[str, str]] = []
    for br in S2_FULL_BRANCHES:
        d = Path("/cmy/cmy/ranAudio/runs/s2_fc_nofc_full") / br
        for f in ("llm_final.safetensors", "projector_final.pt"):
            p = d / f
            if p.is_file():
                s2full_src.append((str(p), f"{br}/{f}"))
        tok = d / "tokenizer"
        if tok.is_dir():
            s2full_src.append((str(tok), f"{br}/tokenizer"))
    s2full_src.append(("/cmy/cmy/ranAudio/LICENSE", "LICENSE"))
    s2full_src.append(("/cmy/cmy/ranAudio/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"))

    out = []
    for repo, srcs, excl in SPECS:
        if repo == "cdsd-moe-lora":
            srcs = cdsd_src
        elif repo == "ranAudio-s15-asr":
            srcs = s15_src
        elif repo == "ranAudio-s2-full":
            srcs = s2full_src
        out.append((repo, srcs, excl))
    SPECS = out


def link_or_copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst.stat().st_size


def should_skip(name: str, extra: set[str]) -> bool:
    if name in TRAIN_JUNK:
        return True
    for pat in extra:
        if pat.startswith("*.") and name.endswith(pat[1:]):
            return True
    return False


def walk_src(src: Path, extra: set[str]):
    """产出 (file, 相对 src 的路径)。目录源递归，文件源直接返回。"""
    if src.is_file():
        yield src, Path(src.name)
        return
    for r, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in TRAIN_JUNK_DIRS and not d.startswith(".")]
        rel_dir = Path(r).relative_to(src)
        for f in files:
            if should_skip(f, extra):
                continue
            if f.startswith("."):
                continue
            yield Path(r) / f, rel_dir / f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    expand_dynamic()

    # 模型卡片：把 modelscope_cards/<repo>/README.md 注入到各发布仓库根目录
    cards_dir = BASE / "ranAudio" / "modelscope_cards"
    for _repo, _srcs, _extra in SPECS:
        _card = cards_dir / _repo / "README.md"
        if _card.exists():
            _srcs.append((str(_card), "README.md"))

    manifest: list[tuple[str, str, int]] = []
    missing: list[str] = []
    grand = 0

    for repo, srcs, extra in SPECS:
        repo_dir = ROOT / repo
        repo_bytes = 0
        repo_files = 0
        for src_str, rel_dst in srcs:
            src = Path(src_str)
            if not src.is_absolute():
                src = BASE / src_str
            if not src.exists():
                missing.append(f"{repo}: {src}")
                continue
            # 源是“实验目录”且有 checkpoint-* 子目录 → 只取最大 step
            if src.is_dir() and any(c.name.startswith("checkpoint-") for c in src.iterdir()):
                cks = sorted(
                    [c for c in src.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")],
                    key=lambda p: int(p.name.rsplit("-", 1)[1]),
                )
                src = cks[-1]
                rel_dst = f"{rel_dst}/{src.name}"
            for f, rel in walk_src(src, extra):
                if args.check:
                    repo_files += 1
                    repo_bytes += f.stat().st_size
                    continue
                # 文件型源：rel_dst 就是目标全路径；目录型源：rel_dst 是前缀
                dst = (repo_dir / rel_dst) if src.is_file() else (repo_dir / rel_dst / rel)
                sz = link_or_copy(f, dst)
                repo_files += 1
                repo_bytes += sz
                manifest.append((repo, str(dst.relative_to(ROOT)), sz))
        grand += repo_bytes
        status = "OK" if repo_bytes else ("EMPTY" if not args.check else "?")
        print(f"{status:<6} {repo:<32} {repo_files:>4} files  {repo_bytes/2**30:>7.2f} GiB")

    print("-" * 62)
    print(f"{'TOTAL':<39} {grand/2**30:>7.2f} GiB")

    if missing:
        print("\n缺失源:")
        for m in missing:
            print("  MISS", m)

    if not args.check:
        ROOT.mkdir(parents=True, exist_ok=True)
        mf = ROOT / "MANIFEST.tsv"
        with mf.open("w", encoding="utf-8") as fh:
            fh.write("repo\tpath\tbytes\n")
            for r, p, s in manifest:
                fh.write(f"{r}\t{p}\t{s}\n")
        print(f"\nMANIFEST: {mf}  ({len(manifest)} 行)")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
