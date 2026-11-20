"""Dataset for gold char-align fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


BLANK_ID = 0


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def char_token_ids(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    """One SP id per character (not EncodeAsIds which may merge)."""
    ids = []
    for ch in text:
        pid = sp.PieceToId(ch)
        if pid <= 0:
            raise ValueError(f"OOV char {ch!r} in {text!r}")
        ids.append(pid)
    return ids


class CharAlignDataset(Dataset):
    def __init__(
        self,
        jsonl: Path,
        tokenizer: str,
        sample_rate: int = 16000,
    ):
        self.rows = load_jsonl(Path(jsonl))
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer)
        self.sample_rate = sample_rate
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        feat = self.fbank(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)  # (T, 80)
        # log-mel like training stack often expects; keep power mel then log
        feat = torch.log(feat.clamp(min=1e-5))
        token_ids = char_token_ids(self.sp, r["text"])
        onsets = [c["onset"] for c in r["chars"]]
        starts = [c["start"] for c in r["chars"]]
        ends = [c["end"] for c in r["chars"]]
        return {
            "id": r["id"],
            "feat": feat,  # (Tf, 80)
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "onsets": torch.tensor(onsets, dtype=torch.float),
            "starts": torch.tensor(starts, dtype=torch.float),
            "ends": torch.tensor(ends, dtype=torch.float),
            "duration": float(r["duration"]),
            "text": r["text"],
            "n_feat": feat.size(0),
        }


def collate_align(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["n_feat"] for b in batch)
    L = max(b["token_ids"].numel() for b in batch)
    Cmax = max(b["onsets"].numel() for b in batch)
    feats = torch.zeros(B, T, 80)
    feat_lens = torch.zeros(B, dtype=torch.long)
    tokens = torch.zeros(B, L, dtype=torch.long)
    token_lens = torch.zeros(B, dtype=torch.long)
    onsets = torch.full((B, Cmax), -1.0)
    starts = torch.full((B, Cmax), -1.0)
    ends = torch.full((B, Cmax), -1.0)
    char_lens = torch.zeros(B, dtype=torch.long)
    ids, texts = [], []
    for i, b in enumerate(batch):
        t = b["n_feat"]
        feats[i, :t] = b["feat"]
        feat_lens[i] = t
        l = b["token_ids"].numel()
        tokens[i, :l] = b["token_ids"]
        token_lens[i] = l
        n = b["onsets"].numel()
        onsets[i, :n] = b["onsets"]
        starts[i, :n] = b["starts"]
        ends[i, :n] = b["ends"]
        char_lens[i] = n
        ids.append(b["id"])
        texts.append(b["text"])
    return {
        "ids": ids,
        "texts": texts,
        "feats": feats,
        "feat_lens": feat_lens,
        "tokens": tokens,
        "token_lens": token_lens,
        "onsets": onsets,
        "starts": starts,
        "ends": ends,
        "char_lens": char_lens,
    }


def build_frame_targets(
    starts: torch.Tensor,
    ends: torch.Tensor,
    char_lens: torch.Tensor,
    token_ids: torch.Tensor,
    enc_lens: torch.Tensor,
    frame_shift_s: float,
    blank_id: int = BLANK_ID,
) -> torch.Tensor:
    """(B, Tenc) long targets: SP id of active char, else blank."""
    B = starts.size(0)
    T = int(enc_lens.max().item())
    T = max(T, 1)
    targets = torch.full((B, T), blank_id, dtype=torch.long, device=starts.device)
    for b in range(B):
        n = int(char_lens[b].item())
        te = int(enc_lens[b].item())
        for t in range(te):
            mid = (t + 0.5) * frame_shift_s
            assigned = blank_id
            for i in range(n):
                if starts[b, i] <= mid < ends[b, i]:
                    assigned = int(token_ids[b, i].item())
                    break
            targets[b, t] = assigned
    return targets


def gaussian_onset_targets(
    onsets: torch.Tensor,
    char_lens: torch.Tensor,
    enc_lens: torch.Tensor,
    frame_shift_s: float,
    sigma_s: float = 0.04,
    time_steps: int | None = None,
) -> torch.Tensor:
    """(B, T) soft onset peaks in [0,1]."""
    B = onsets.size(0)
    T = int(time_steps) if time_steps is not None else int(enc_lens.max().item())
    T = max(T, 1)
    device = onsets.device
    t_idx = torch.arange(T, device=device).float()
    centers = (t_idx + 0.5) * frame_shift_s  # (T,)
    out = torch.zeros(B, T, device=device)
    sig2 = 2 * sigma_s * sigma_s
    for b in range(B):
        n = int(char_lens[b].item())
        te = min(int(enc_lens[b].item()), T)
        peaks = []
        for i in range(n):
            o = onsets[b, i]
            if o < 0:
                continue
            peaks.append(torch.exp(-((centers[:te] - o) ** 2) / sig2))
        if peaks:
            stacked = torch.stack(peaks, dim=0).max(dim=0).values
            out[b, :te] = stacked.clamp(0, 1)
    return out
