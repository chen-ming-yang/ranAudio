"""Stream real-asr Lhotse WenetSpeech / AISHELL manifests. Do not ingest Zipformer fbank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import librosa
import numpy as np

from ranAudio.paths import AISHELL_MANIFEST, WENET_FLAC, WENET_MANIFEST, WENET_S_JSONL


@dataclass
class Segment:
    sid: str
    wav: Path
    start: float
    duration: float
    text: str
    subsets: tuple[str, ...] = ()


def load_recordings(path: Path) -> dict[str, Path]:
    """id -> audio path. Recordings.jsonl is small enough to keep in RAM."""
    out: dict[str, Path] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sources = row.get("sources") or []
            src = sources[0]["source"] if sources else ""
            if src:
                out[str(row["id"])] = Path(src)
    return out


def iter_supervisions(
    path: Path,
    *,
    recordings: dict[str, Path],
    subsets: Optional[set[str]] = None,
    require_audio: bool = False,
) -> Iterator[Segment]:
    """Yield one supervision at a time. Does not slurp the 1.5M-line file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            custom = row.get("custom") or {}
            row_subsets = tuple(custom.get("subsets") or [])
            if subsets and not (set(row_subsets) & subsets):
                continue
            rid = str(row.get("recording_id") or "")
            wav = recordings.get(rid)
            if wav is None:
                continue
            if require_audio and not wav.is_file():
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            yield Segment(
                sid=str(row.get("id") or ""),
                wav=wav,
                start=float(row.get("start") or 0.0),
                duration=float(row.get("duration") or 0.0),
                text=text,
                subsets=row_subsets,
            )


def write_offset_index(jsonl: Path, index_path: Path) -> int:
    offsets: list[int] = []
    with jsonl.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, np.asarray(offsets, dtype=np.int64))
    return len(offsets)


def load_offset_index(index_path: Path) -> np.ndarray:
    return np.load(index_path)


def read_jsonl_at(jsonl: Path, offset: int) -> dict[str, Any]:
    with jsonl.open("rb") as f:
        f.seek(int(offset))
        return json.loads(f.readline())


def resolve_audio_path(wav: Path) -> Path:
    """Prefer 16k flac (fast seek) over opus. Same layout as real-asr convert_opus_to_flac."""
    wav = Path(wav)
    s = str(wav)
    marker = "/audio/train/"
    if marker in s:
        flac = Path(s.replace("/DATA_UNTAR_DIRECTORY/audio/train/", "/flac_16k/train/"))
        flac = flac.with_suffix(".flac")
        if flac.is_file():
            return flac
    if WENET_FLAC.is_dir() and marker in s:
        rel = s.split(marker, 1)[1]
        flac = (WENET_FLAC / "train" / rel).with_suffix(".flac")
        if flac.is_file():
            return flac
    alt = wav.with_suffix(".flac")
    if alt.is_file():
        return alt
    return wav


def load_segment_wav(seg: Segment, sr: int = 16000) -> np.ndarray:
    """Crop by start/duration. soundfile seek on flac; librosa fallback for opus."""
    path = resolve_audio_path(seg.wav)
    start = max(0.0, float(seg.start))
    dur = max(0.05, float(seg.duration))
    try:
        import soundfile as sf

        with sf.SoundFile(str(path), "r") as f:
            file_sr = int(f.samplerate)
            start_fr = min(int(start * file_sr), max(0, len(f) - 1))
            n_fr = max(1, int(dur * file_sr))
            f.seek(start_fr)
            data = f.read(frames=n_fr, dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=-1)
        wav = np.asarray(data, dtype=np.float32).reshape(-1)
        if file_sr != sr and wav.size:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        return wav
    except Exception:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav, _ = librosa.load(
                str(path),
                sr=sr,
                mono=True,
                offset=start,
                duration=dur,
            )
        return np.asarray(wav, dtype=np.float32).reshape(-1)


def export_compact(
    supervisions: Path,
    recordings: Path,
    out_jsonl: Path,
    *,
    subsets: Optional[set[str]] = None,
    max_samples: int = 0,
    require_audio: bool = True,
) -> int:
    recs = load_recordings(recordings)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as w:
        for seg in iter_supervisions(
            supervisions, recordings=recs, subsets=subsets, require_audio=require_audio
        ):
            rec = {
                "id": seg.sid,
                "wav": str(seg.wav),
                "start": seg.start,
                "duration": seg.duration,
                "text": seg.text,
                "subsets": list(seg.subsets),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if max_samples and n >= max_samples:
                break
    return n


def compact_to_segment(row: dict[str, Any]) -> Segment:
    return Segment(
        sid=str(row.get("id") or ""),
        wav=Path(row["wav"]),
        start=float(row.get("start") or 0.0),
        duration=float(row.get("duration") or 0.0),
        text=str(row.get("text") or ""),
        subsets=tuple(row.get("subsets") or ()),
    )


def iter_compact(jsonl: Path) -> Iterator[Segment]:
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield compact_to_segment(json.loads(line))


def default_split_paths(name: str) -> tuple[Path, Path]:
    """name: wenet-train | wenet-ms | wenet-dev | wenet-test_net | aishell-test | aishell-train"""
    if name == "wenet-train":
        root = WENET_MANIFEST / "train"
    elif name == "wenet-ms":
        root = WENET_MANIFEST / "train_ms"
    elif name == "wenet-dev":
        root = WENET_MANIFEST / "dev"
    elif name == "wenet-test_net":
        root = WENET_MANIFEST / "test_net"
    elif name == "aishell-test":
        root = AISHELL_MANIFEST / "test"
    elif name == "aishell-train":
        root = AISHELL_MANIFEST / "train"
    else:
        raise ValueError(f"unknown split {name}")
    return root / "supervisions.jsonl", root / "recordings.jsonl"


class IndexedCompact:
    """Random-access compact jsonl via byte offsets. Rows stay on disk."""

    def __init__(self, jsonl: Path, index_path: Optional[Path] = None):
        self.jsonl = Path(jsonl)
        idx = Path(index_path) if index_path else self.jsonl.with_suffix(".offsets.npy")
        if not idx.is_file():
            write_offset_index(self.jsonl, idx)
        self.offsets = load_offset_index(idx)

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def __getitem__(self, i: int) -> Segment:
        return compact_to_segment(read_jsonl_at(self.jsonl, int(self.offsets[i])))


def export_s_default(out: Path = WENET_S_JSONL, max_samples: int = 0) -> int:
    sup, rec = default_split_paths("wenet-ms")
    return export_compact(sup, rec, out, subsets={"S"}, max_samples=max_samples, require_audio=True)


class CompactWavDataset:
    """Worker-side wav crop. Rows stay on disk via IndexedCompact."""

    def __init__(
        self,
        indexed: IndexedCompact,
        indices: list[int],
        *,
        min_dur: float = 0.5,
        max_dur: float = 16.0,
    ):
        self.indexed = indexed
        self.indices: list[int] = []
        self.durations: list[float] = []
        for i in indices:
            seg = indexed[i]
            if min_dur <= seg.duration <= max_dur:
                self.indices.append(i)
                self.durations.append(float(seg.duration))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Optional[dict[str, Any]]:
        seg = self.indexed[self.indices[i]]
        try:
            wav = load_segment_wav(seg)
        except Exception:
            return None
        if wav.size < 1600:
            return None
        return {"wav": wav, "text": seg.text, "duration": float(seg.duration), "sid": seg.sid}


def collate_wav_batch(batch: list[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
    rows = [b for b in batch if b is not None]
    if not rows:
        return None
    return {
        "wavs": [r["wav"] for r in rows],
        "texts": [r["text"] for r in rows],
        "durations": [r["duration"] for r in rows],
        "sids": [r["sid"] for r in rows],
    }


class DurationBatchSampler:
    """Pack similar-length clips until max_batch_dur, like real-asr bucketing."""

    def __init__(
        self,
        durations: list[float],
        max_batch_dur: float,
        *,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.durations = durations
        self.max_batch_dur = max(0.5, float(max_batch_dur))
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def __iter__(self):
        import random

        order = list(range(len(self.durations)))
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            rng.shuffle(order)
        order.sort(key=lambda i: self.durations[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        tot = 0.0
        for i in order:
            d = self.durations[i]
            if cur and tot + d > self.max_batch_dur:
                batches.append(cur)
                cur, tot = [], 0.0
            cur.append(i)
            tot += d
        if cur and not self.drop_last:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        tot = 0.0
        n = 0
        for d in sorted(self.durations):
            if tot and tot + d > self.max_batch_dur:
                n += 1
                tot = 0.0
            tot += d
        if tot and not self.drop_last:
            n += 1
        return n
