"""Load real-asr Zipformer CTC + optional onset head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REAL_ASR_ROOT = Path("/cmy/cmy/real-asr")
if str(REAL_ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_ASR_ROOT))

from train.train import build_model, load_config  # noqa: E402


class OnsetHead(nn.Module):
    def __init__(self, encoder_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, 1)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        # (B, T, 1) logits
        return self.proj(encoder_out).squeeze(-1)


def load_asr_bundle(
    config: str = "/cmy/cmy/real-asr/configs/train_finetune_dysarthric.yaml",
    checkpoint: str = "/cmy/cmy/real-asr/exp/finetune_dysarthric/epoch_30.pt",
    device: str = "cuda",
):
    cfg = load_config(config)
    transducer, ctc_head = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transducer.load_state_dict(ckpt["model"], strict=False)
    ctc_head.load_state_dict(ckpt["ctc_head"], strict=False)
    transducer = transducer.to(device)
    ctc_head = ctc_head.to(device)
    onset_head = OnsetHead(encoder_dim=cfg["model"].get("encoder_output_dim", 512)).to(device)
    return cfg, transducer, ctc_head, onset_head


@torch.no_grad()
def encoder_frame_shift_s(transducer, device="cuda", n_feat_frames: int = 800) -> float:
    """Estimate encoder seconds/frame from a dummy forward (fbank hop=10ms)."""
    x = torch.zeros(1, n_feat_frames, 80, device=device)
    lens = torch.tensor([n_feat_frames], device=device)
    enc, elens = transducer.encoder(x, lens)
    t_out = int(elens[0].item())
    # fbank hop 10ms → duration = n_feat_frames * 0.01
    return (n_feat_frames * 0.01) / max(t_out, 1)


def forced_align_onsets(
    log_probs: torch.Tensor,
    token_ids: torch.Tensor,
    enc_len: int,
    frame_shift_s: float,
    blank_id: int = 0,
) -> list[float]:
    """CTC forced-align → per-token onset (first non-blank frame mid time).

    log_probs: (T, V)  token_ids: (L,)
    """
    import torchaudio

    T = int(enc_len)
    lp = log_probs[:T].unsqueeze(0)  # (1,T,V)
    tgt = token_ids.unsqueeze(0)
    in_lens = torch.tensor([T], device=log_probs.device)
    tgt_lens = torch.tensor([token_ids.numel()], device=log_probs.device)
    try:
        alignments, _scores = torchaudio.functional.forced_align(
            lp, tgt, in_lens, tgt_lens, blank=blank_id
        )
        # alignments: (1, T) label per frame
        ali = alignments[0, :T].tolist()
    except Exception:
        # greedy fallback: walk tokens
        ali = _greedy_monotonic_align(log_probs[:T], token_ids, blank_id)

    onsets = []
    L = token_ids.numel()
    seen = 0
    prev = blank_id
    for t, lab in enumerate(ali):
        if lab == blank_id:
            prev = blank_id
            continue
        if lab != prev:
            # new token emission
            if seen < L and lab == int(token_ids[seen].item()):
                onsets.append((t + 0.5) * frame_shift_s)
                seen += 1
            prev = lab
        else:
            prev = lab
    # if under-aligned, pad with last
    while len(onsets) < L:
        onsets.append(onsets[-1] if onsets else 0.0)
    return onsets[:L]


def forced_align_spans(
    log_probs: torch.Tensor,
    token_ids: torch.Tensor,
    enc_len: int,
    blank_id: int = 0,
) -> list[tuple[int, int]]:
    """CTC forced-align → per-token emission span [start_frame, end_frame] (closed).

    Returns list of (start, end) frame indices, one per token in token_ids.
    Spans are separated by blank runs, so they don't overlap.
    """
    import torchaudio

    T = int(enc_len)
    lp = log_probs[:T].unsqueeze(0)  # (1,T,V)
    tgt = token_ids.unsqueeze(0)
    in_lens = torch.tensor([T], device=log_probs.device)
    tgt_lens = torch.tensor([token_ids.numel()], device=log_probs.device)
    try:
        alignments, _scores = torchaudio.functional.forced_align(
            lp, tgt, in_lens, tgt_lens, blank=blank_id
        )
        ali = alignments[0, :T].tolist()
    except Exception:
        ali = _greedy_monotonic_align(log_probs[:T], token_ids, blank_id)

    L = token_ids.numel()
    spans: list[tuple[int, int]] = []
    cur = None  # (token_idx, start_frame)
    for t, lab in enumerate(ali):
        if lab == blank_id:
            if cur is not None:
                spans.append((cur[1], t - 1))
                cur = None
            continue
        # non-blank token
        if cur is None:
            # new emission of the next token in sequence
            if len(spans) < L:
                cur = (len(spans), t)
        else:
            # continuing or switching token (repeated char case)
            prev_idx = cur[0]
            if lab != token_ids[min(prev_idx, L - 1)].item():
                spans.append((cur[1], t - 1))
                if len(spans) < L:
                    cur = (len(spans), t)
                else:
                    cur = None
    if cur is not None:
        spans.append((cur[1], T - 1))
    while len(spans) < L:
        spans.append((T - 1, T - 1))
    return spans[:L]


def _greedy_monotonic_align(log_probs, token_ids, blank_id=0):
    """Simple DP-free monotonic: at each step pick blank or next token."""
    T, V = log_probs.shape
    L = token_ids.numel()
    ali = [blank_id] * T
    j = 0
    for t in range(T):
        if j >= L:
            ali[t] = blank_id
            continue
        tid = int(token_ids[j].item())
        # prefer emitting next token if competitive
        if log_probs[t, tid] >= log_probs[t, blank_id]:
            ali[t] = tid
            j += 1
        else:
            ali[t] = blank_id
    return ali


def onset_peaks_onsets(
    onset_logits: torch.Tensor,
    n_chars: int,
    enc_len: int,
    frame_shift_s: float,
    min_gap_s: float = 0.05,
) -> list[float]:
    """Peak-pick from onset-head logits → per-char onset (top-N by score, time-ordered).

    onset_logits: (T,) raw logits. n_chars known from the forced text.
    min_gap_s: suppress duplicate picks of the same broad peak (greedy NMS).
    """
    T = int(enc_len)
    probs = torch.sigmoid(onset_logits[:T]).detach().cpu().numpy().astype(float)
    if T < 3:
        return [0.0] * n_chars
    # local maxima (strict rise, non-strict fall for plateaus)
    cand: list[tuple[float, int]] = []
    for t in range(1, T - 1):
        if probs[t] > probs[t - 1] and probs[t] >= probs[t + 1]:
            cand.append((probs[t], t))
    # greedy NMS by score, then order by time
    cand.sort(key=lambda x: -x[0])
    min_gap_frames = max(int(round(min_gap_s / frame_shift_s)), 1)
    picked: list[tuple[float, int]] = []
    for score, t in cand:
        if all(abs(t - p) >= min_gap_frames for _, p in picked):
            picked.append((score, t))
        if len(picked) >= n_chars:
            break
    picked.sort(key=lambda x: x[1])
    onsets = [(t + 0.5) * frame_shift_s for _, t in picked]
    while len(onsets) < n_chars:
        onsets.append(onsets[-1] if onsets else 0.0)
    return onsets[:n_chars]


def onset_monotonic_dp(
    onset_logits: torch.Tensor,
    n_chars: int,
    enc_len: int,
    frame_shift_s: float,
    min_gap_frames: int = 1,
) -> list[float]:
    """Monotonic DP peak selection: pick N strictly-increasing onset frames maximizing
    sum of onset probability. Enforces one onset per char and time ordering, unlike top-N.

    onset_logits: (T,). n_chars known from the forced text.
    """
    import numpy as np

    T = int(enc_len)
    probs = torch.sigmoid(onset_logits[:T]).detach().cpu().numpy().astype(float)
    if n_chars <= 0:
        return []
    if T < n_chars:
        return [(t + 0.5) * frame_shift_s for t in range(n_chars)]
    NEG = -1e9
    dp = np.full((n_chars, T), NEG, dtype=float)
    back = np.full((n_chars, T), -1, dtype=np.int64)
    dp[0] = probs.copy()
    for i in range(1, n_chars):
        best_prev = NEG
        best_idx = -1
        for t in range(T):
            if t - min_gap_frames >= 0:
                idx = t - min_gap_frames
                if dp[i - 1, idx] > best_prev:
                    best_prev = dp[i - 1, idx]
                    best_idx = idx
            if best_idx >= 0:
                dp[i, t] = best_prev + probs[t]
                back[i, t] = best_idx
    last_t = int(np.argmax(dp[n_chars - 1]))
    idxs = [0] * n_chars
    idxs[n_chars - 1] = last_t
    for i in range(n_chars - 1, 0, -1):
        idxs[i - 1] = int(back[i, idxs[i]])
    return [(t + 0.5) * frame_shift_s for t in idxs]
