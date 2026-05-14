"""Merge every synthetic Q&A file into one deduplicated training set.

Reads from ``data/synthetic/``:
    * ``qa_v2_dataset.jsonl``   — instruction/output schema, current pipeline
    * ``qa_v3_dataset.jsonl``   — instruction/output schema, prior pipeline run
    * ``qa_gemini.jsonl``       — alternate generator output
    * ``qa_claude.jsonl``       — alternate generator output
    * ``qa_manual_ar.jsonl``    — hand-curated Arabic seeds
    * ``qa_manual_en.jsonl``    — hand-curated English seeds

Normalises to ``{instruction, output, language, source}``, dedupes by the
lowercased+stripped instruction string, and writes
``data/synthetic/qa_train_all.jsonl``.

Usage:
    uv run python -m src.training.merge_datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

OUTPUT_FILE: Path = settings.paths.synthetic / "qa_train_all.jsonl"

# Source → (path, default_source_tag).
SOURCES: list[tuple[str, str]] = [
    ("qa_v2_dataset.jsonl", "claude_v2"),
    ("qa_v3_dataset.jsonl", "claude_v3"),
    ("qa_gemini.jsonl", "gemini"),
    ("qa_claude.jsonl", "claude_curated"),
    ("qa_manual_ar.jsonl", "manual_ar"),
    ("qa_manual_en.jsonl", "manual_en"),
]


def _detect_language(text: str) -> str:
    """Cheap language tag: 'ar' if any Arabic codepoint present, else 'en'."""
    for ch in text:
        if "؀" <= ch <= "ۿ":
            return "ar"
    return "en"


def _normalise(row: dict[str, Any], default_source: str) -> dict[str, Any] | None:
    """Return ``{instruction, output, language, source}`` or ``None`` if unusable."""
    instruction = row.get("instruction") or row.get("question") or ""
    output = row.get("output") or row.get("answer") or ""

    instruction = (instruction or "").strip()
    output = (output or "").strip()

    if not instruction or not output:
        return None
    if len(instruction) < 5 or len(output) < 3:
        return None

    # Prefer the row's own source tag if it has one.
    source = row.get("source") or default_source

    # Prefer the row's own language/stream tag when present.
    stream = row.get("stream", "")
    if stream in ("ar_msa", "ar_saudi"):
        language = "ar"
    elif stream == "en":
        language = "en"
    else:
        language = _detect_language(instruction)

    return {
        "instruction": instruction,
        "output": output,
        "language": language,
        "source": source,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, ignoring blank / malformed lines."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        logger.warning("missing %s", path)
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("%s line %d: bad JSON (%s)", path.name, i, exc)
    return rows


def merge() -> dict[str, Any]:
    """Read every source file, dedupe, write the merged JSONL. Returns stats."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    per_source_raw: Counter[str] = Counter()
    per_source_kept: Counter[str] = Counter()
    per_language: Counter[str] = Counter()

    for fname, default_source in SOURCES:
        path = settings.paths.synthetic / fname
        raw = _read_jsonl(path)
        per_source_raw[default_source] = len(raw)

        for row in raw:
            norm = _normalise(row, default_source)
            if norm is None:
                continue
            key = norm["instruction"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(norm)
            per_source_kept[norm["source"]] += 1
            per_language[norm["language"]] += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_merged": len(merged),
        "raw_counts": dict(per_source_raw),
        "kept_by_source": dict(per_source_kept),
        "by_language": dict(per_language),
        "output_file": str(OUTPUT_FILE),
    }
    logger.info("Merge stats: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge every synthetic Q&A file into one deduped JSONL."
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    _parse_args(argv)
    stats = merge()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _main()
