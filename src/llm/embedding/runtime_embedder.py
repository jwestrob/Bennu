from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeEmbedderConfig:
    model_name: str
    window_size: int
    overlap: int
    aggregation: str  # 'mean' | 'max' | 'weighted_mean'
    device: str = "cpu"


class ESM2RuntimeEmbedder:
    """
    Deterministic runtime embedder for protein sequences.

    Mirrors pipeline settings from embedding_manifest.json to avoid drift.
    Requires transformers + torch at runtime. No randomness; model in eval mode.
    """

    def __init__(self, cfg: RuntimeEmbedderConfig):
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Runtime ESM2 embedder requires transformers+torch installed"
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self._model = AutoModel.from_pretrained(cfg.model_name)
        self._model.eval()
        self._device = self._select_device(cfg.device)
        self._model.to(self._device)
        self.cfg = cfg
        self.embedding_dim = getattr(self._model.config, "hidden_size", None) or 0
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise RuntimeError("Unable to determine embedding dimension from model.config.hidden_size")

    def _select_device(self, device_spec: str):
        torch = self._torch
        if device_spec == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def load_manifest(manifest_path: Path) -> RuntimeEmbedderConfig:
        data = json.loads(Path(manifest_path).read_text())
        model_name = data.get("model_name") or data.get("model") or "facebook/esm2_t33_650M_UR50D"
        window_size = int(data.get("window_size", 1022))
        overlap = int(data.get("overlap", 128))
        aggregation = str(data.get("aggregation_strategy", "mean"))
        return RuntimeEmbedderConfig(
            model_name=model_name,
            window_size=window_size,
            overlap=overlap,
            aggregation=aggregation,
            device=os.getenv("AGENT_EMBEDDER_DEVICE", "cpu"),
        )

    def _embed_chunk(self, seq: str):
        torch = self._torch
        with torch.no_grad():
            tokens = self._tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            outputs = self._model(**tokens)
            # Mean pool over sequence length dimension
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            emb = last_hidden.mean(dim=1).squeeze(0)  # [H]
            return emb.detach().cpu().numpy()

    def embed_sequence(self, sequence: str):
        seq = (sequence or "").strip().upper()
        if not seq:
            raise ValueError("Empty sequence for embedding")

        # Sliding windows with overlap per pipeline
        win = max(1, int(self.cfg.window_size))
        ov = max(0, int(self.cfg.overlap))
        if len(seq) <= win:
            return self._embed_chunk(seq)

        chunks = []
        i = 0
        while i < len(seq):
            chunk = seq[i : i + win]
            if not chunk:
                break
            chunks.append(self._embed_chunk(chunk))
            if i + win >= len(seq):
                break
            i += max(1, win - ov)

        # Aggregate
        import numpy as np
        if not chunks:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        mat = np.vstack(chunks)
        agg = self.cfg.aggregation.lower()
        if agg == "max":
            vec = np.max(mat, axis=0)
        elif agg in ("weighted_mean", "weighted-average"):
            # Simple position-based weighting as placeholder for pipeline parity
            w = np.linspace(1.0, 2.0, num=mat.shape[0], dtype=np.float32)
            w = w / w.sum()
            vec = (mat * w[:, None]).sum(axis=0)
        else:  # mean
            vec = np.mean(mat, axis=0)
        return vec.astype(np.float32)


def find_embedding_manifest(lancedb_path: str | Path) -> Optional[Path]:
    base = Path(lancedb_path)
    candidates = [
        base / "embedding_manifest.json",
        base.parent / "embedding_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

