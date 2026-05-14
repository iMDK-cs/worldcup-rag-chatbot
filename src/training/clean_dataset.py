"""Filter the merged training set to remove obviously-bad rows.

Concrete failure modes we strip out (observed in the live chatbot):

    1. Non-WC2026 Arab teams hallucinated as participants
       (Oman, Bahrain, Kuwait, UAE, Lebanon, …).
    2. Outputs that lead with banned prose like "بناءً على" / "وفقًا لـ"
       / "Based on" / "According to" — the model echoes these.
    3. Outputs mentioning fictitious rules ("قانون الـ X ثوانٍ لرميات
       التماس") that some generator runs produced.
    4. Outputs with the prompt's banned phrases anywhere in the body.
    5. Trivial/garbage rows (output < 5 chars, instruction < 5 chars,
       output equal to instruction).
    6. Duplicate ``instruction`` strings (case-insensitive).

Read:  ``data/synthetic/qa_train_all.jsonl``
Write: ``data/synthetic/qa_train_clean.jsonl``
Also: ``data/synthetic/qa_train_clean_report.json`` with per-rule counts.

Usage:
    uv run python -m src.training.clean_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

INPUT_FILE: Path = settings.paths.synthetic / "qa_train_all.jsonl"
OUTPUT_FILE: Path = settings.paths.synthetic / "qa_train_clean.jsonl"
REPORT_FILE: Path = settings.paths.synthetic / "qa_train_clean_report.json"

# Arab countries NOT in the 2026 World Cup. Any pair that mentions one of
# these as a *participating team* is almost certainly hallucinated.
NOT_IN_WC2026: tuple[str, ...] = (
    "عمان", "Oman",
    "البحرين", "Bahrain",
    "الكويت", "Kuwait",
    "الإمارات", "UAE", "United Arab Emirates",
    "لبنان", "Lebanon",
    "اليمن", "Yemen",
    "سوريا", "Syria",
    "ليبيا", "Libya",
    "السودان", "Sudan",
    "موريتانيا", "Mauritania",
    "فلسطين", "Palestine",
)

# Suspicious phrases — when an answer talks about World Cup probabilities,
# matches, or groups while naming a non-participant, drop it.
PARTICIPATION_HINTS: tuple[str, ...] = (
    "مجموعة", "Group", "احتمال", "probability", "مباراة", "match",
    "يلعب", "plays", "ضد", "vs ", "تواجه",
)

BANNED_PREFIXES: tuple[str, ...] = (
    "بناءً على", "بناء على", "وفقاً لـ", "وفقا لـ",
    "Based on", "According to", "As an AI", "I'm sorry",
)

# Patterns that signal fabricated football rules. The "قانون الـ N ثوانٍ"
# pattern from the bad chat session is the prime example.
HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"قانون\s+الـ?\s*\d+\s+ثوان"),    # "law of N seconds"
    re.compile(r"rule of \d+ seconds", re.IGNORECASE),
    re.compile(r"قاعدة\s+\d+\s+ثوان"),
)


def _violation(row: dict[str, Any]) -> str | None:
    """Return a reason string if the row should be dropped, else ``None``."""
    instr = (row.get("instruction") or "").strip()
    out = (row.get("output") or "").strip()

    if len(instr) < 5 or len(out) < 5:
        return "too_short"
    if instr.lower() == out.lower():
        return "instruction_equals_output"

    for bad in BANNED_PREFIXES:
        if out.lower().lstrip().startswith(bad.lower()):
            return f"banned_prefix:{bad}"
        if instr.lower().lstrip().startswith(bad.lower()):
            return f"banned_prefix_in_instruction:{bad}"

    for pat in HALLUCINATION_PATTERNS:
        if pat.search(out):
            return "hallucinated_rule_pattern"

    body = instr + " " + out
    body_lower = body.lower()
    for country in NOT_IN_WC2026:
        if country in body:
            # Only flag when it's framed as a participant.
            if any(hint in body for hint in PARTICIPATION_HINTS) or \
               any(hint.lower() in body_lower for hint in PARTICIPATION_HINTS):
                return f"non_participant:{country}"

    return None


def clean() -> dict[str, Any]:
    """Read INPUT_FILE → write OUTPUT_FILE with bad rows removed. Returns stats."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}. Run merge_datasets first."
        )

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()

    total = 0
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            reasons["bad_json"] += 1
            continue

        # Dedup by lowercased instruction.
        key = (row.get("instruction") or "").strip().lower()
        if key in seen:
            reasons["duplicate_instruction"] += 1
            continue

        why = _violation(row)
        if why is not None:
            # Bucket reasons by their prefix for nicer reporting.
            bucket = why.split(":", 1)[0]
            reasons[bucket] += 1
            continue

        seen.add(key)
        kept.append(row)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats: dict[str, Any] = {
        "input_rows": total,
        "kept": len(kept),
        "dropped": total - len(kept),
        "drop_reasons": dict(reasons),
        "output_file": str(OUTPUT_FILE),
    }
    REPORT_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cleaning stats:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Drop hallucinated / banned rows from the merged training set."
    ).parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    _parse_args(argv)
    stats = clean()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _main()
