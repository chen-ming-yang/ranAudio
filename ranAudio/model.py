"""MiniCPM5-2B + SenseVoice-Small encoder (default) or Whisper APM + projector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from ranAudio.paths import MINICPM5, MINICPMO, SENSEVOICE
from ranAudio.sensevoice_apm import SenseVoiceAPM
from ranAudio.tools import fc_system_prompt, format_tool_xml

AUDIO_START = "<|audio_start|>"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_END = "<|audio_end|>"
AUDIO_TOKENS = [AUDIO_START, AUDIO_PAD, AUDIO_END]
POOL_STEP = 5


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(audio_features)))


def _whisper_config_from_minicpmo(path: Path) -> WhisperConfig:
    raw = json.loads((path / "config.json").read_text())
    ac = dict(raw["audio_config"])
    keep = (
        "d_model",
        "encoder_layers",
        "encoder_attention_heads",
        "encoder_ffn_dim",
        "num_mel_bins",
        "max_source_positions",
        "dropout",
        "activation_function",
        "attention_dropout",
        "activation_dropout",
        "encoder_layerdrop",
        "scale_embedding",
        "init_std",
    )
    kwargs = {k: ac[k] for k in keep if k in ac}
    kwargs["model_type"] = "whisper"
    return WhisperConfig(**kwargs)


def _load_apm_from_minicpmo(path: Path, device: torch.device, dtype: torch.dtype) -> WhisperEncoder:
    cfg = _whisper_config_from_minicpmo(path)
    enc = WhisperEncoder(cfg)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted({v for k, v in index["weight_map"].items() if k.startswith("apm.")})
    state: dict[str, torch.Tensor] = {}
    from safetensors import safe_open

    for shard in shards:
        with safe_open(str(path / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("apm."):
                    state[k[4:]] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if unexpected:
        print(f"apm unexpected keys={unexpected[:8]}", flush=True)
    if missing:
        print(f"apm missing keys={missing[:8]} n={len(missing)}", flush=True)
    return enc.to(device=device, dtype=dtype)


def _feat_lengths(mel_len: int, pool_step: int = POOL_STEP) -> tuple[int, int]:
    after_cnn = (mel_len - 1) // 2 + 1
    after_pool = (after_cnn - pool_step) // pool_step + 1
    return after_cnn, after_pool


class MiniCPM5Audio(nn.Module):
    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer,
        apm: nn.Module,
        projector: MultiModalProjector,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        pool_step: int = POOL_STEP,
        encoder_backend: str = "sensevoice",
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.apm = apm
        self.projector = projector
        self.feature_extractor = feature_extractor
        self.pool_step = pool_step
        self.encoder_backend = encoder_backend
        self.audio_start_id = tokenizer.convert_tokens_to_ids(AUDIO_START)
        self.audio_pad_id = tokenizer.convert_tokens_to_ids(AUDIO_PAD)
        self.audio_end_id = tokenizer.convert_tokens_to_ids(AUDIO_END)

    @property
    def device(self) -> torch.device:
        return next(self.llm.parameters()).device

    @property
    def hidden_size(self) -> int:
        return int(self.llm.config.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        llm_path: str | Path = MINICPM5,
        apm_path: str | Path = MINICPMO,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        projector_path: Optional[str | Path] = None,
        encoder_path: Optional[str | Path] = None,
        encoder_backend: str = "sensevoice",
        sensevoice_path: str | Path = SENSEVOICE,
    ) -> "MiniCPM5Audio":
        llm_path = Path(llm_path)
        apm_path = Path(apm_path)
        tok = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True)
        added = tok.add_special_tokens({"additional_special_tokens": AUDIO_TOKENS})
        llm = AutoModelForCausalLM.from_pretrained(
            str(llm_path),
            dtype=dtype,
            trust_remote_code=True,
        )
        if added > 0:
            llm.resize_token_embeddings(len(tok))
        llm.to(device)
        fe = None
        pool_step = POOL_STEP
        if encoder_backend == "whisper":
            apm = _load_apm_from_minicpmo(apm_path, device=torch.device(device), dtype=dtype)
            in_dim = int(apm.config.d_model)
            fe = WhisperFeatureExtractor(
                feature_size=80,
                sampling_rate=16000,
                hop_length=160,
                n_fft=400,
                padding_value=0.0,
                return_attention_mask=False,
            )
        elif encoder_backend == "sensevoice":
            apm = SenseVoiceAPM(
                sensevoice_path, device=torch.device(device), dtype=torch.float32
            )
            in_dim = int(apm.out_dim)
            pool_step = 1
        else:
            raise ValueError(f"unknown encoder_backend={encoder_backend}")
        projector = MultiModalProjector(in_dim, int(llm.config.hidden_size)).to(device=device, dtype=dtype)
        if projector_path:
            try:
                projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
            except RuntimeError as exc:
                print(f"skip projector {projector_path}: {exc}", flush=True)
        if encoder_path:
            enc_state = torch.load(encoder_path, map_location=device, weights_only=True)
            apm.load_state_dict(enc_state)
        model = cls(
            llm,
            tok,
            apm,
            projector,
            fe,
            pool_step=pool_step,
            encoder_backend=encoder_backend,
        )
        print(
            f"MiniCPM5Audio backend={encoder_backend} hidden={model.hidden_size} "
            f"enc_d={in_dim} added_tokens={added} vocab={len(tok)}",
            flush=True,
        )
        return model

    def encode_wav(self, wav: np.ndarray) -> torch.Tensor:
        """Return pooled audio embeddings [T, H]."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if self.encoder_backend == "sensevoice":
            return self.encode_wavs([wav])[0]
        feats = self.feature_extractor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )["input_features"].to(device=self.device, dtype=next(self.apm.parameters()).dtype)
        # Stock WhisperEncoder requires 3000 mel frames; crop back to the real CNN length.
        mel_len = int(feats.shape[-1])
        expected = int(self.apm.config.max_source_positions) * 2
        if feats.shape[-1] < expected:
            feats = torch.nn.functional.pad(feats, (0, expected - feats.shape[-1]))
        elif feats.shape[-1] > expected:
            feats = feats[..., :expected]
            mel_len = expected
        _, t_pool = _feat_lengths(mel_len, self.pool_step)
        after_cnn, _ = _feat_lengths(mel_len, self.pool_step)
        hidden = self.apm(feats).last_hidden_state[:, :after_cnn, :]
        embeds = self.projector(hidden)
        embeds = embeds.transpose(1, 2)
        embeds = nn.functional.avg_pool1d(embeds, self.pool_step, stride=self.pool_step)
        embeds = embeds.transpose(1, 2)
        t = min(t_pool, int(embeds.shape[1]))
        return embeds[0, :t, :]

    def encode_wavs(self, wavs: list[np.ndarray]) -> list[torch.Tensor]:
        """Encode a list of waveforms to projector embeddings [T_i, H]."""
        if self.encoder_backend == "sensevoice":
            hiddens = self.apm.encode_wavs([np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs])
            proj_dtype = next(self.projector.parameters()).dtype
            out: list[torch.Tensor] = []
            for hidden in hiddens:
                embeds = self.projector(hidden.to(dtype=proj_dtype))
                if self.pool_step > 1:
                    x = embeds.transpose(0, 1).unsqueeze(0)
                    x = nn.functional.avg_pool1d(x, self.pool_step, stride=self.pool_step)
                    embeds = x.transpose(1, 2)[0]
                out.append(embeds)
            return out
        return [self.encode_wav(w) for w in wavs]

    def _apply_chat(
        self,
        *,
        n_audio: int = 0,
        tools: Optional[list[dict]] = None,
        assistant: Optional[str] = None,
        add_generation_prompt: bool = False,
        system: Optional[str] = None,
        user_text: Optional[str] = None,
    ) -> str:
        if user_text is not None:
            user_content = user_text
        else:
            user_content = AUDIO_START + (AUDIO_PAD * n_audio) + AUDIO_END
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or fc_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            kwargs["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("tools", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _embed_ids(self, input_ids: torch.Tensor, audio_embeds: torch.Tensor) -> torch.Tensor:
        tok_embeds = self.llm.get_input_embeddings()(input_ids)
        pad_mask = input_ids[0] == self.audio_pad_id
        n_pad = int(pad_mask.sum().item())
        if n_pad != audio_embeds.shape[0]:
            raise RuntimeError(f"audio pad tokens={n_pad} != audio frames={audio_embeds.shape[0]}")
        tok_embeds = tok_embeds.clone()
        tok_embeds[0, pad_mask] = audio_embeds.to(tok_embeds.dtype)
        return tok_embeds

    def teacher_force_batch(
        self,
        wav: np.ndarray,
        assistant: str,
        tools: Optional[list[dict]] = None,
        system: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        audio_embeds = self.encode_wav(wav)
        n = int(audio_embeds.shape[0])
        full = self._apply_chat(
            n_audio=n, tools=tools, assistant=assistant, add_generation_prompt=False, system=system
        )
        prefix = self._apply_chat(
            n_audio=n, tools=tools, assistant=None, add_generation_prompt=True, system=system
        )
        if not full.startswith(prefix):
            raise RuntimeError("chat template prefix/full mismatch")
        full_ids = self.tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        prefix_len = int(prefix_ids.shape[1])
        if full_ids[0, :prefix_len].tolist() != prefix_ids[0].tolist():
            # token-level mismatch (normalization); locate by pad span then suffix
            prefix_len = int(prefix_ids.shape[1])
        labels = full_ids.clone()
        labels[:, :prefix_len] = -100
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[full_ids == pad_id] = -100
        inputs_embeds = self._embed_ids(full_ids, audio_embeds)
        attn = torch.ones_like(full_ids)
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attn,
            "labels": labels,
            "n_label_tokens": int((labels != -100).sum().item()),
            "prefix_len": prefix_len,
        }

    def teacher_force_many(
        self,
        wavs: list[np.ndarray],
        assistants: list[str],
        tools: Optional[list[dict]] = None,
        system: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Batched teacher force. Pads to the longest chat+audio sequence in the batch."""
        audio_list = self.encode_wavs(wavs)
        rows_emb: list[torch.Tensor] = []
        rows_lab: list[torch.Tensor] = []
        n_label = 0
        pad_id = self.tokenizer.pad_token_id
        for audio, assistant in zip(audio_list, assistants):
            n = int(audio.shape[0])
            full = self._apply_chat(
                n_audio=n, tools=tools, assistant=assistant, add_generation_prompt=False, system=system
            )
            prefix = self._apply_chat(
                n_audio=n, tools=tools, assistant=None, add_generation_prompt=True, system=system
            )
            full_ids = self.tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
                self.device
            )
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
                self.device
            )
            prefix_len = int(prefix_ids.shape[0])
            labels = full_ids.clone()
            labels[:prefix_len] = -100
            if pad_id is not None:
                labels[full_ids == pad_id] = -100
            tok_embeds = self.llm.get_input_embeddings()(full_ids.unsqueeze(0))[0].clone()
            pad_mask = full_ids == self.audio_pad_id
            if int(pad_mask.sum().item()) != audio.shape[0]:
                raise RuntimeError(
                    f"audio pad tokens={int(pad_mask.sum().item())} != audio frames={audio.shape[0]}"
                )
            tok_embeds[pad_mask] = audio.to(tok_embeds.dtype)
            rows_emb.append(tok_embeds)
            rows_lab.append(labels)
            n_label += int((labels != -100).sum().item())
        max_t = max(e.shape[0] for e in rows_emb)
        bsz = len(rows_emb)
        hid = rows_emb[0].shape[-1]
        inputs = rows_emb[0].new_zeros(bsz, max_t, hid)
        lab_pad = torch.full((bsz, max_t), -100, device=self.device, dtype=torch.long)
        attn = torch.zeros(bsz, max_t, device=self.device, dtype=torch.long)
        for i, (emb, lab) in enumerate(zip(rows_emb, rows_lab)):
            t = emb.shape[0]
            inputs[i, :t] = emb
            lab_pad[i, :t] = lab
            attn[i, :t] = 1
        return {
            "inputs_embeds": inputs,
            "attention_mask": attn,
            "labels": lab_pad,
            "n_label_tokens": n_label,
            "batch_size": bsz,
        }

    @torch.no_grad()
    def generate_text(
        self,
        wav: np.ndarray,
        tools: Optional[list[dict]] = None,
        system: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        self.eval()
        audio_embeds = self.encode_wav(wav)
        n = int(audio_embeds.shape[0])
        prefix = self._apply_chat(
            n_audio=n, tools=tools, assistant=None, add_generation_prompt=True, system=system
        )
        input_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        inputs_embeds = self._embed_ids(input_ids, audio_embeds)
        attn = torch.ones_like(input_ids)
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # generate with inputs_embeds returns only new tokens on some versions, full on others
        if out.shape[1] > input_ids.shape[1] and torch.equal(out[:, : input_ids.shape[1]], input_ids):
            new_ids = out[:, input_ids.shape[1] :]
        else:
            new_ids = out
        # MiniCPM5 marks <function / <param as special tokens; skipping them
        # would leave only ` name="enable_bluetooth">` and break XML parse.
        return self._decode_new(new_ids)

    def _decode_new(self, new_ids: torch.Tensor) -> str:
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=False)
        drop = list(AUDIO_TOKENS) + [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            self.tokenizer.eos_token or "",
            self.tokenizer.bos_token or "",
            self.tokenizer.pad_token or "",
        ]
        for tok in drop:
            if tok:
                text = text.replace(tok, "")
        return text.strip()

    @torch.no_grad()
    def generate_from_text(
        self,
        user_text: str,
        tools: Optional[list[dict]] = None,
        system: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Text-only generate (no audio pads). Used for bare-2B FC / T2T probes."""
        self.eval()
        prefix = self._apply_chat(
            user_text=user_text,
            tools=tools,
            assistant=None,
            add_generation_prompt=True,
            system=system,
        )
        input_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        attn = torch.ones_like(input_ids)
        out = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if out.shape[1] > input_ids.shape[1] and torch.equal(out[:, : input_ids.shape[1]], input_ids):
            new_ids = out[:, input_ids.shape[1] :]
        else:
            new_ids = out
        return self._decode_new(new_ids)


def freeze_except_projector(model: MiniCPM5Audio) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def freeze_llm_train_encoder(model: MiniCPM5Audio) -> list[nn.Parameter]:
    """S1.5: 2B frozen (no LoRA); encoder + projector trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.apm.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def save_projector(model: MiniCPM5Audio, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), path)


def save_encoder(model: MiniCPM5Audio, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.apm.state_dict(), path)


def load_encoder(model: MiniCPM5Audio, path: str | Path) -> None:
    state = torch.load(path, map_location=model.device, weights_only=True)
    model.apm.load_state_dict(state)
