"""构音障碍音色克隆 — 效果评估。

- 可懂度: Qwen3-ASR 转写 + CER
- 说话人相似度: FunASR cam++（cosine similarity）
- 对比报告: 零样本 vs 微调

运行环境: wenet-asr-compare（qwen_asr + jiwer）
    PYTHONPATH=/cmy/cmy/Qwen3-ASR python -m ranAudio.voice_clone.evaluate --spk 06

用法:
    PYTHONPATH=/cmy/cmy/Qwen3-ASR python -m ranAudio.voice_clone.evaluate --spk 06
    PYTHONPATH=/cmy/cmy/Qwen3-ASR python -m ranAudio.voice_clone.evaluate --spk 06 --modes xvec icl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ranAudio.voice_clone.config import PILOT_SPK, QWEN3_TTS_ROOT, ref_dir, spk_dir, zs_out_dir
from ranAudio.voice_clone.utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

QWEN3_ASR_ROOT = Path("/cmy/cmy/Qwen3-ASR")
QWEN3_ASR_MODEL = QWEN3_ASR_ROOT / "Qwen3-ASR-1.7B"

# 确保 qwen_asr 可导入
if str(QWEN3_ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_ASR_ROOT))


# ── CER 计算 ──────────────────────────────────────────

def compute_cer(ref: str, hyp: str) -> float:
    """计算字错误率（CER）。"""
    import jiwer

    if not ref:
        return 1.0 if hyp else 0.0
    return jiwer.cer(ref, hyp)


def norm_text(text: str) -> str:
    """规范化文本：去空白、标点。"""
    import re

    text = re.sub(r"[\s，。、；：？！,.!?;:]+", "", text)
    return text.strip()


# ── 说话人相似度 ──────────────────────────────────────

def cosine_similarity(a, b) -> float:
    import numpy as np

    a, b = np.array(a).flatten(), np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def extract_speaker_embedding_funasr(model, wav_path: str):
    """FunASR cam++ 提取说话人嵌入。"""
    import numpy as np

    import torch

    res = model.generate(input=wav_path, output_dir=None)
    if isinstance(res, list) and len(res) > 0:
        emb = res[0].get("spk_embedding")
        if emb is None:
            emb = res[0].get("embedding")
        if emb is not None:
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            return np.array(emb).flatten()
    raise ValueError(f"无法提取说话人嵌入: {wav_path}")


# ── Qwen3-ASR 转写 ────────────────────────────────────

class Qwen3ASRTranscriber:
    """Qwen3-ASR 转写器。"""

    def __init__(self, model_path: str | Path | None = None, device: str = "cuda:0"):
        self.model_path = str(model_path or QWEN3_ASR_MODEL)
        self.device = device
        self.wrapper = None
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        logger.info("加载 Qwen3-ASR: %s", self.model_path)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.wrapper = Qwen3ASRModel.from_pretrained(self.model_path, dtype=dtype, device_map=None)
        self.model = self.wrapper.model
        self.processor = self.wrapper.processor
        logger.info("Qwen3-ASR 加载完成")

    def transcribe(self, wav_path: str, max_new_tokens: int = 128) -> str:
        """转写单条音频。"""
        import librosa
        import torch

        assert self.model is not None, "请先调用 load()"

        # 加载音频为 numpy (16kHz mono)
        wav, _ = librosa.load(wav_path, sr=16000, mono=True)

        # 构建 prompt（与 cdsd/moe_eval_cer.py 一致）
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        prompt = self.processor.apply_chat_template(
            [msgs], add_generation_prompt=True, tokenize=False
        )[0]

        inputs = self.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=dtype)

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        if hasattr(gen_out, "sequences"):
            seqs = gen_out.sequences[:, prompt_len:]
        else:
            seqs = gen_out[:, prompt_len:]
        text = self.processor.batch_decode(seqs, skip_special_tokens=True)[0]
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[-1]
        return text.strip()


# ── 主评估流程 ────────────────────────────────────────

def run_evaluation(
    spk: str = PILOT_SPK,
    modes: list[str] | None = None,
    max_items: int = 50,
) -> dict:
    """执行评估，返回指标字典。"""
    if modes is None:
        modes = ["xvec"]

    # 加载参考音频信息
    ref_candidates = read_jsonl(ref_dir(spk) / "ref_candidates.jsonl")
    ref_wav = ref_candidates[0]["wav"]

    # 加载 Qwen3-ASR
    asr = Qwen3ASRTranscriber()
    asr.load()

    # 加载说话人验证模型（可选）
    sv_model = None
    ref_emb = None
    try:
        from funasr import AutoModel

        logger.info("加载说话人验证模型 (cam++)...")
        sv_model = AutoModel(model="cam++")
        ref_emb = extract_speaker_embedding_funasr(sv_model, ref_wav)
        logger.info("参考说话人嵌入提取完成")
    except ImportError:
        logger.warning("funasr 未安装，跳过说话人相似度评估")

    # 加载测试集（目标文本 + 原始音频）
    test_items = read_jsonl(spk_dir(spk) / "finetune_test.jsonl")[:max_items]

    all_results = {}

    for mode in modes:
        if mode == "finetuned":
            out_dir = spk_dir(spk) / "finetuned"
        else:
            out_dir = zs_out_dir(spk, mode)

        # zeroshot 写 results.jsonl；demo.py 写 demo_results.jsonl
        results_file = out_dir / "results.jsonl"
        if not results_file.exists() and (out_dir / "demo_results.jsonl").exists():
            results_file = out_dir / "demo_results.jsonl"
        if not results_file.exists():
            logger.warning("模式 %s 无结果文件，跳过: %s", mode, results_file)
            continue

        gen_results = read_jsonl(results_file)[:max_items]

        mode_metrics = {
            "mode": mode,
            "n_samples": len(gen_results),
            "items": [],
            "avg_spk_sim": 0.0,
            "avg_cer": 0.0,
            "avg_orig_cer": 0.0,
        }

        spk_sims = []
        cers = []
        orig_cers = []

        for i, gen in enumerate(gen_results):
            gen_wav = gen["wav"]
            # 结果文件自带 text 时优先使用（demo.py 输出与测试集顺序不一定对齐）
            orig = test_items[i] if i < len(test_items) else {}
            ref_text = gen.get("text") or orig.get("text", "")

            # 说话人相似度
            spk_sim = None
            if sv_model is not None:
                try:
                    gen_emb = extract_speaker_embedding_funasr(sv_model, gen_wav)
                    spk_sim = cosine_similarity(ref_emb, gen_emb)
                except Exception as e:
                    logger.warning("说话人嵌入提取失败 %s: %s", gen_wav, e)

            # 生成音频 CER（Qwen3-ASR）
            try:
                hyp_text = asr.transcribe(gen_wav)
                cer = compute_cer(norm_text(ref_text), norm_text(hyp_text))
            except Exception as e:
                logger.warning("ASR 转写失败 %s: %s", gen_wav, e)
                hyp_text = ""
                cer = 1.0

            # 原始构音音频 CER（对照基线）
            orig_cer = None
            try:
                orig_wav = orig.get("audio", orig.get("wav", ""))
                if orig_wav and Path(orig_wav).exists():
                    orig_hyp = asr.transcribe(orig_wav)
                    orig_cer = compute_cer(norm_text(ref_text), norm_text(orig_hyp))
            except Exception:
                pass

            item_result = {
                "id": gen.get("id", f"item_{i}"),
                "text": ref_text,
                "hyp_text": hyp_text,
                "gen_wav": gen_wav,
                "spk_sim": round(spk_sim, 4) if spk_sim is not None else None,
                "cer": round(cer, 4),
                "orig_cer": round(orig_cer, 4) if orig_cer is not None else None,
            }
            mode_metrics["items"].append(item_result)
            if spk_sim is not None:
                spk_sims.append(spk_sim)
            cers.append(cer)
            if orig_cer is not None:
                orig_cers.append(orig_cer)

            sim_str = f"{spk_sim:.3f}" if spk_sim is not None else "N/A"
            logger.info("[%s %d/%d] spk_sim=%s cer=%.3f", mode, i + 1, len(gen_results), sim_str, cer)

        if spk_sims:
            mode_metrics["avg_spk_sim"] = round(sum(spk_sims) / len(spk_sims), 4)
        if cers:
            mode_metrics["avg_cer"] = round(sum(cers) / len(cers), 4)
        if orig_cers:
            mode_metrics["avg_orig_cer"] = round(sum(orig_cers) / len(orig_cers), 4)

        all_results[mode] = mode_metrics

    # 生成报告
    report_path = spk_dir(spk) / "eval_report.md"
    generate_report(all_results, spk, report_path)

    # 保存 JSON 结果
    json_path = spk_dir(spk) / "eval_results.jsonl"
    flat_items = []
    for mode, metrics in all_results.items():
        for item in metrics["items"]:
            flat_items.append({"mode": mode, **item})
    write_jsonl(json_path, flat_items)

    logger.info("评估完成，报告: %s", report_path)
    return all_results


def generate_report(results: dict, spk: str, out_path: Path) -> None:
    """生成 Markdown 对比报告。"""
    lines = [
        f"# 构音障碍音色克隆评估报告 — S{spk}",
        "",
        "评估时间: 2026-11-03",
        "ASR 引擎: Qwen3-ASR-1.7B",
        "",
        "## 总体指标",
        "",
        "| 模式 | 样本数 | 说话人相似度↑ | CER↓ | 原始构音 CER↓ |",
        "|------|--------|-------------|------|-------------|",
    ]

    for mode, m in results.items():
        sim_str = f"{m['avg_spk_sim']:.4f}" if m.get("avg_spk_sim") else "N/A"
        orig_cer_str = f"{m['avg_orig_cer']:.4f}" if m.get("avg_orig_cer") else "N/A"
        lines.append(
            f"| {mode} | {m['n_samples']} | {sim_str} | {m['avg_cer']:.4f} | {orig_cer_str} |"
        )

    lines += [
        "",
        "## 指标说明",
        "",
        "- **说话人相似度**: cam++ 说话人嵌入余弦相似度（与参考音频对比），越高越好",
        "- **CER**: 合成语音 Qwen3-ASR 转写 vs 目标文本的字错误率，越低越好",
        "- **原始构音 CER**: 原始构音障碍音频的 Qwen3-ASR 转写 CER（对照基线）",
        "",
        "## 详细结果",
        "",
    ]

    for mode, m in results.items():
        lines.append(f"### 模式: {mode}")
        lines.append("")
        lines.append("| ID | 目标文本 | ASR 转写 | 说话人相似度 | CER | 原始CER |")
        lines.append("|----|---------|---------|------------|-----|---------|")
        for item in m["items"][:20]:
            sim_str = f"{item['spk_sim']:.3f}" if item.get("spk_sim") is not None else "N/A"
            orig_cer_str = f"{item['orig_cer']:.3f}" if item.get("orig_cer") is not None else "N/A"
            lines.append(
                f"| {item['id']} | {item['text']} | {item['hyp_text']} | {sim_str} | {item['cer']:.3f} | {orig_cer_str} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已生成: %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构音障碍音色克隆 — 效果评估（Qwen3-ASR）")
    parser.add_argument("--spk", type=str, default=PILOT_SPK)
    parser.add_argument("--modes", nargs="+", default=["xvec"], help="评估模式列表")
    parser.add_argument("--max-items", type=int, default=50)
    args = parser.parse_args()

    results = run_evaluation(spk=args.spk, modes=args.modes, max_items=args.max_items)

    print("\n=== 评估结果 ===")
    for mode, m in results.items():
        sim_str = f"{m['avg_spk_sim']:.4f}" if m.get("avg_spk_sim") else "N/A"
        print(f"  [{mode}] spk_sim={sim_str}, cer={m['avg_cer']:.4f}")


if __name__ == "__main__":
    main()
