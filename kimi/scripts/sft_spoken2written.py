#!/usr/bin/env python3
"""Text LoRA SFT: CS2W spoken → written on Qwen3.5 (default 0.8B).

Env: prefer conda env `qwen3` (transformers>=5.x + peft).
Data: eval_spoken2written/sft/{train,val}.jsonl from prepare_sft_spoken2written.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

KIMI = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/cmy/cmy/Qwen/Qwen3.5-0.8B")
DEFAULT_DATA = KIMI / "eval_spoken2written" / "sft"
DEFAULT_OUT = KIMI / "eval_spoken2written" / "sft_out" / "qwen35_0.8b_lora"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _build_labels(tokenizer, messages, max_len: int) -> dict:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = [-100] * len(input_ids)
    assistant_content = messages[-1]["content"]
    assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
    seq_len = len(input_ids)
    ast_len = len(assistant_tokens)
    matched = False
    if ast_len > 0 and seq_len >= ast_len:
        for start in range(seq_len - ast_len, -1, -1):
            if input_ids[start : start + ast_len] == assistant_tokens:
                labels[start : start + ast_len] = assistant_tokens
                matched = True
                break
    if input_ids and input_ids[-1] == tokenizer.eos_token_id:
        labels[-1] = tokenizer.eos_token_id
    n_supervised = sum(1 for x in labels if x != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "matched": matched,
        "n_supervised": n_supervised,
        "seq_len": seq_len,
    }


class SFTDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with data_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line)["messages"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        out = _build_labels(self.tokenizer, self.samples[idx], self.max_len)
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": out["labels"],
        }


def sanity_check_labels(tokenizer, data_path: Path, max_len: int, n_check: int = 100) -> None:
    matched = 0
    max_seen = 0
    n = 0
    with data_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_check:
                break
            messages = json.loads(line)["messages"]
            out = _build_labels(tokenizer, messages, max_len)
            max_seen = max(max_seen, out["seq_len"])
            if out["matched"] and out["n_supervised"] > 1:
                matched += 1
            n += 1
    if n == 0:
        raise RuntimeError(f"empty dataset: {data_path}")
    rate = matched / n
    print(f"label align: {matched}/{n} = {rate:.1%}; max_seq={max_seen}/{max_len}")
    if rate < 0.95:
        raise RuntimeError(f"label align too low ({rate:.1%})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("SFT_MODEL", str(DEFAULT_MODEL))))
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SFT_DATA_DIR", str(DEFAULT_DATA))))
    ap.add_argument("--output-dir", type=Path, default=Path(os.environ.get("SFT_OUTPUT_DIR", str(DEFAULT_OUT))))
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("SFT_EPOCHS", "2")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("SFT_BATCH", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("SFT_GRAD_ACCUM", "2")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("SFT_LR", "2e-4")))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("SFT_MAX_LEN", "512")))
    ap.add_argument("--lora-r", type=int, default=int(os.environ.get("SFT_LORA_R", "16")))
    ap.add_argument("--lora-alpha", type=int, default=int(os.environ.get("SFT_LORA_ALPHA", "32")))
    ap.add_argument("--lora-dropout", type=float, default=float(os.environ.get("SFT_LORA_DROPOUT", "0.05")))
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    if not train_path.is_file():
        print(f"ERROR missing {train_path}; run prepare_sft_spoken2written.py first", file=sys.stderr)
        return 1
    if not args.model.is_dir():
        print(f"ERROR model not found: {args.model}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model:  {args.model}")
    print(f"train:  {train_path}")
    print(f"val:    {val_path if val_path.is_file() else '(none)'}")
    print(f"output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sanity_check_labels(tokenizer, train_path, args.max_seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_path, tokenizer, args.max_seq_len)
    eval_ds = SFTDataset(val_path, tokenizer, args.max_seq_len) if val_path.is_file() else None
    print(f"train n={len(train_ds)}" + (f"  val n={len(eval_ds)}" if eval_ds else ""))

    eff = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // eff)
    warmup_steps = max(10, int(args.epochs * steps_per_epoch * args.warmup_ratio))

    targs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        gradient_checkpointing=True,
    )
    if args.max_steps > 0:
        targs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**{k: v for k, v in targs.items() if v is not None})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt"
        ),
        processing_class=tokenizer,
    )

    meta = {
        "model": str(args.model),
        "train": str(train_path),
        "val": str(val_path) if eval_ds else None,
        "n_train": len(train_ds),
        "n_val": len(eval_ds) if eval_ds else 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
    }
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== start SFT ===")
    trainer.train()
    print(f"\nsaving → {args.output_dir}")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
