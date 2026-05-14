"""RAG pipeline: ChromaDB index over World Cup CSVs + synthetic Q&A.

Builds a single multilingual collection that mixes:
    * Structured CSV rows (schedule, host cities, match probabilities) rendered
      as natural-language sentences so a multilingual embedder can match them
      against either Arabic or English questions.
    * The 1060 synthetic Q&A pairs as ``"Q: ... A: ..."`` chunks.

Embedding model: ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
(supports Arabic + English in the same vector space, small/fast).

Storage: persisted Chroma collection at ``data/chroma_db/``.

Usage:
    # one-time index build
    uv run python -m src.rag.pipeline --build

    # programmatic retrieval
    from src.rag.pipeline import retrieve
    hits = retrieve("Who plays in Group A?", top_k=5)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME: str = "worldcup2026"
CHROMA_DIR_NAME: str = "chroma_db"
CHROMA_PERSIST_DIR: Path = settings.paths.data / CHROMA_DIR_NAME

# The embedder is small and CPU-fast; keep it off the GPU so it never fights
# the fine-tuning job for VRAM.
EMBED_DEVICE: str = "cpu"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    """One retrieved chunk with its metadata and similarity score."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def _schedule_rows() -> list[tuple[str, dict[str, str]]]:
    """Render the FIFA2026 schedule CSV into natural-language sentences."""
    path = settings.paths.csvs / "FIFA2026_schedule.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return []

    rows: list[tuple[str, dict[str, str]]] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            match = r["match_number"]
            group = r["group"]
            stadium = r["stadium"]
            date = r["date"]
            text = (
                f"{match} of the FIFA World Cup 2026 ({group}) is scheduled "
                f"for {date} at {stadium}."
            )
            rows.append(
                (
                    text,
                    {
                        "source": "schedule",
                        "match": match,
                        "group": group,
                        "stadium": stadium,
                        "date": date,
                    },
                )
            )
    return rows


def _host_city_rows() -> list[tuple[str, dict[str, str]]]:
    """Render the host_cities CSV into descriptive sentences."""
    path = settings.paths.csvs / "host_cities.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return []

    rows: list[tuple[str, dict[str, str]]] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = (
                f"{r['city_name']}, {r['country']} hosts FIFA World Cup 2026 "
                f"matches at {r['venue_name']} (airport {r['airport_code']}, "
                f"{r['region_cluster']} region)."
            )
            rows.append(
                (
                    text,
                    {
                        "source": "host_cities",
                        "city": r["city_name"],
                        "country": r["country"],
                        "venue": r["venue_name"],
                        "region": r["region_cluster"],
                    },
                )
            )
    return rows


def _probability_rows() -> list[tuple[str, dict[str, str]]]:
    """Render the match-probabilities CSV into prose."""
    path = settings.paths.csvs / "future_match_probabilities_baseline.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return []

    rows: list[tuple[str, dict[str, str]]] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                p_home = float(r["p_home_win"])
                p_draw = float(r["p_draw"])
                p_away = float(r["p_away_win"])
            except (ValueError, KeyError):
                continue
            text = (
                f"In Group {r['group']}, the baseline model gives "
                f"{r['home_team']} a {p_home:.1%} chance of beating "
                f"{r['away_team']}, with a {p_draw:.1%} chance of a draw and "
                f"a {p_away:.1%} chance for {r['away_team']} to win."
            )
            rows.append(
                (
                    text,
                    {
                        "source": "probabilities",
                        "group": r["group"],
                        "home_team": r["home_team"],
                        "away_team": r["away_team"],
                        "p_home_win": f"{p_home:.4f}",
                        "p_draw": f"{p_draw:.4f}",
                        "p_away_win": f"{p_away:.4f}",
                    },
                )
            )
    return rows


def _qa_rows() -> list[tuple[str, dict[str, str]]]:
    """Treat each synthetic Q&A pair as a knowledge chunk.

    Prefers the merged ``qa_train_all.jsonl`` (instruction/output schema)
    when present. Falls back to the legacy ``qa_arabic_all.jsonl`` +
    ``qa_english_all.jsonl`` pair otherwise.
    """
    rows: list[tuple[str, dict[str, str]]] = []

    # Prefer the cleaned file if it's present (filtered for hallucinations);
    # otherwise fall back to the raw merged file.
    clean = settings.paths.synthetic / "qa_train_clean.jsonl"
    merged = clean if clean.exists() else (settings.paths.synthetic / "qa_train_all.jsonl")
    if merged.exists():
        with merged.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                question = rec.get("instruction") or rec.get("question") or ""
                answer = rec.get("output") or rec.get("answer") or ""
                if not question or not answer:
                    continue
                lang = rec.get("language", "en")
                rows.append(
                    (
                        f"Q: {question}\nA: {answer}",
                        {
                            "source": f"qa_{lang}",
                            "language": lang,
                            "origin": rec.get("source", "synthetic"),
                        },
                    )
                )
        logger.info("Loaded %d Q&A chunks from %s (merged)", len(rows), merged)
        return rows

    for fname, lang in (
        ("qa_english_all.jsonl", "en"),
        ("qa_arabic_all.jsonl", "ar"),
    ):
        path = settings.paths.synthetic / fname
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = f"Q: {rec['question']}\nA: {rec['answer']}"
                rows.append(
                    (
                        text,
                        {
                            "source": f"qa_{lang}",
                            "language": lang,
                            "origin": rec.get("source", "synthetic"),
                        },
                    )
                )
    return rows


def _collect_documents() -> list[tuple[str, dict[str, str]]]:
    """Gather every chunk we want indexed."""
    docs: list[tuple[str, dict[str, str]]] = []
    docs.extend(_schedule_rows())
    docs.extend(_host_city_rows())
    docs.extend(_probability_rows())
    docs.extend(_qa_rows())
    logger.info("Collected %d documents to index.", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Chroma + embedding handles
# ---------------------------------------------------------------------------

_embed_model: Any = None
_collection: Any = None


def _get_embed_model() -> Any:
    """Lazy-load the multilingual sentence-transformer."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedder %s on %s …", EMBED_MODEL_NAME, EMBED_DEVICE)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    return _embed_model


def _embed(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts to plain Python lists (Chroma's expected format)."""
    model = _get_embed_model()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def _get_collection(create_if_missing: bool = True) -> Any:
    """Open (or create) the persisted Chroma collection."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if create_if_missing:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Build / retrieve
# ---------------------------------------------------------------------------

def build_index(rebuild: bool = False) -> int:
    """Embed every document and write it into the Chroma collection.

    Args:
        rebuild: If True, drop the existing collection first.

    Returns:
        Number of documents indexed.
    """
    import chromadb

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Dropped existing collection '%s'.", COLLECTION_NAME)
        except Exception:
            pass
        global _collection
        _collection = None

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs = _collect_documents()
    if not docs:
        raise RuntimeError("No documents collected — check the source CSVs/JSONLs.")

    texts = [t for t, _ in docs]
    metadatas = [m for _, m in docs]
    ids = [f"doc-{i:06d}" for i in range(len(texts))]

    batch = 256
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        embeddings = _embed(texts[start:end])
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        logger.info("Indexed %d / %d", end, len(texts))

    logger.info(
        "Chroma collection '%s' now has %d documents at %s",
        COLLECTION_NAME,
        collection.count(),
        CHROMA_PERSIST_DIR,
    )
    return collection.count()


def retrieve(query: str, top_k: int = 5) -> list[RetrievalHit]:
    """Retrieve the top-k most similar chunks for ``query``.

    Returns ``RetrievalHit`` records (text, source, score, metadata) sorted by
    descending similarity. Score is ``1 - cosine_distance`` (higher = closer).
    """
    collection = _get_collection(create_if_missing=False)
    query_vec = _embed([query])[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = results.get("documents") or [[]]
    meta_list = results.get("metadatas") or [[]]
    dist_list = results.get("distances") or [[]]

    documents = docs_list[0] if docs_list else []
    metadatas = meta_list[0] if meta_list else []
    distances = dist_list[0] if dist_list else []

    hits: list[RetrievalHit] = []
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        hits.append(
            RetrievalHit(
                text=text,
                source=str(meta.get("source", "unknown")),
                score=float(1.0 - dist),
                metadata=dict(meta),
            )
        )
    return hits


def collection_size() -> int:
    """Number of indexed documents (raises if the collection is missing)."""
    return int(_get_collection(create_if_missing=False).count())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the World Cup RAG index.")
    parser.add_argument("--build", action="store_true", help="(Re)build the Chroma index.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and rebuild from scratch."
    )
    parser.add_argument("--query", type=str, default=None, help="Run a single retrieval.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.build or args.rebuild:
        n = build_index(rebuild=args.rebuild)
        print(f"Indexed {n} documents into '{COLLECTION_NAME}'.")

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
