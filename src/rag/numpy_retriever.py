"""Numpy-backed multilingual retriever (no chromadb / sentence-transformers).

Why this exists: on Windows, ``chromadb`` and ``sentence-transformers`` native
libraries segfault when imported in the same process as a 4-bit bitsandbytes
model. The FastAPI server hits that combination on every request, so we
sidestep both libraries entirely:

    * Index format: ``data/chroma_db/numpy_index/`` containing
      ``embeddings.npy`` (float32, L2-normalised, shape ``[N, D]``),
      ``texts.json``, and ``metadatas.json``.
    * Encoder: raw ``transformers`` MiniLM (mean-pooled + L2-normalised) —
      bit-equivalent to ``sentence-transformers`` for the same checkpoint, but
      without ST's native wrapper.
    * Retrieval: cosine similarity as a single ``embeddings @ q`` matmul.

Usage:
    uv run python -m src.rag.numpy_retriever --build      # build the index
    uv run python -m src.rag.numpy_retriever --query "…"  # smoke-test

    from src.rag.numpy_retriever import retrieve
    hits = retrieve("Who plays in Group A?", top_k=5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR: Path = settings.paths.data / "chroma_db" / "numpy_index"
EMBEDDINGS_PATH: Path = INDEX_DIR / "embeddings.npy"
TEXTS_PATH: Path = INDEX_DIR / "texts.json"
METADATAS_PATH: Path = INDEX_DIR / "metadatas.json"

EMBED_DEVICE: str = "cpu"  # tiny model; keep off the GPU for VRAM hygiene
ENCODE_BATCH: int = 32
MAX_LEN: int = 256



# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    """One retrieved chunk with its metadata and cosine score."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Encoder (raw transformers, mean-pool + L2 normalise)
# ---------------------------------------------------------------------------

_tokenizer: Any = None
_encoder: Any = None


def _load_encoder() -> tuple[Any, Any]:
    """Lazy-load tokenizer + MiniLM via plain transformers."""
    global _tokenizer, _encoder
    if _tokenizer is None or _encoder is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading encoder %s on %s …", EMBED_MODEL_NAME, EMBED_DEVICE)
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True)
        _encoder = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _encoder.to(EMBED_DEVICE)
        _encoder.eval()
    return _tokenizer, _encoder


def _encode(texts: list[str]) -> np.ndarray:
    """Tokenise → forward → mean-pool by attention mask → L2-normalise.

    Returns a ``[len(texts), hidden_dim]`` ``float32`` numpy array. The
    encoder is forced to fp32 on every call because loading a 4-bit
    bitsandbytes model in the same process can silently down-cast other
    nn.Modules in memory (observed on Windows).
    """
    import torch
    import torch.nn.functional as F

    tok, model = _load_encoder()
    model.float()  # defensive: re-assert fp32 weights every call
    outputs: list[np.ndarray] = []

    for start in range(0, len(texts), ENCODE_BATCH):
        batch = texts[start : start + ENCODE_BATCH]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(EMBED_DEVICE) for k, v in enc.items()}
        with torch.no_grad(), torch.amp.autocast(device_type="cpu", enabled=False):
            out = model(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state.float() * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normed = F.normalize(pooled, p=2, dim=1)
        outputs.append(normed.cpu().to(torch.float32).numpy())

    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 384), dtype=np.float32)


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

_index_embeddings: np.ndarray | None = None
_index_texts: list[str] | None = None
_index_metadatas: list[dict[str, Any]] | None = None


def _load_index() -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Load embeddings + texts + metadatas from disk (lazy, cached)."""
    global _index_embeddings, _index_texts, _index_metadatas

    if _index_embeddings is None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Numpy index missing at {INDEX_DIR}. Build it with "
                f"`python -m src.rag.numpy_retriever --build`."
            )
        logger.info("Loading numpy index from %s …", INDEX_DIR)
        _index_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        _index_texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
        _index_metadatas = json.loads(METADATAS_PATH.read_text(encoding="utf-8"))
        logger.info(
            "Index: %d docs, dim=%d", _index_embeddings.shape[0], _index_embeddings.shape[1]
        )

    assert _index_texts is not None and _index_metadatas is not None
    return _index_embeddings, _index_texts, _index_metadatas


# ---------------------------------------------------------------------------
# Build / retrieve
# ---------------------------------------------------------------------------

def build_index() -> int:
    """Rebuild the numpy index from the same documents pipeline.py uses.

    Imports ``src.rag.pipeline`` lazily because that module is a peer.
    """
    from src.rag.pipeline import _collect_documents

    docs = _collect_documents()
    if not docs:
        raise RuntimeError("No documents collected — check the source CSVs/JSONLs.")

    texts = [t for t, _ in docs]
    metadatas = [m for _, m in docs]

    logger.info("Encoding %d documents …", len(texts))
    embeddings = _encode(texts)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    TEXTS_PATH.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    METADATAS_PATH.write_text(json.dumps(metadatas, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Wrote %d embeddings (dim=%d) to %s",
        embeddings.shape[0],
        embeddings.shape[1],
        INDEX_DIR,
    )
    return embeddings.shape[0]


def retrieve(query: str, top_k: int = 5) -> list[RetrievalHit]:
    """Top-k cosine-similarity retrieval against the numpy index."""
    embeddings, texts, metadatas = _load_index()

    query_vec = _encode([query])[0]  # already L2-normalised → cosine == dot
    scores = embeddings @ query_vec  # [N]

    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    hits: list[RetrievalHit] = []
    for i in top_idx:
        meta = metadatas[i] or {}
        hits.append(
            RetrievalHit(
                text=texts[i],
                source=str(meta.get("source", "unknown")),
                score=float(scores[i]),
                metadata=dict(meta),
            )
        )
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the numpy RAG index.")
    parser.add_argument("--build", action="store_true", help="(Re)build the numpy index.")
    parser.add_argument("--query", type=str, default=None, help="Run a single retrieval.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.build:
        n = build_index()
        print(f"Built numpy index with {n} documents at {INDEX_DIR}.")

    if args.query:
        hits = retrieve(args.query, top_k=args.top_k)
        print(f"\nTop-{args.top_k} hits for: {args.query!r}\n")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h.score:.3f}  source={h.source}")
            print(f"    {h.text}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _main()
