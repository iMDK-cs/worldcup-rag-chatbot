"""Minimal benchmark over the running chat API.

Runs a fixed battery of queries against ``/chat/stream``, measures:

    * ``ttft``    — time to first answer token (perceived latency)
    * ``total``   — total request duration (server + network)
    * ``hit``     — whether the answer contains any of the ``expect`` keywords
    * ``route``   — non-LLM fast-path (social/static/refusal) vs LLM

Writes the per-query rows + a summary block (p50 / p95 / mean / hit-rate by
category and overall) to ``models/benchmark_report.json``.

Usage:
    uv run python -m src.evaluation.benchmark
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from src.config import settings

API_URL: str = "http://localhost:8000/chat/stream"
OUTPUT_FILE: Path = settings.paths.models / "benchmark_report.json"
TIMEOUT_S: int = 120

# (category, language, message, expected_substrings_any_of)
QUERIES: list[tuple[str, str, str, list[str]]] = [
    # social / identity
    ("social", "ar", "كيف حالك",       ["بخير", "جاهز"]),
    ("social", "ar", "مين انت",        ["Mundial"]),
    ("social", "ar", "السلام عليكم",   ["وعليكم", "Mundial"]),
    ("social", "en", "who are you",    ["Mundial"]),

    # static facts (historical / format)
    ("static", "ar", "كم كأس عالم عند البرازيل",       ["5", "1958"]),
    ("static", "ar", "كم ألقاب فرنسا",                  ["2", "1998"]),
    ("static", "ar", "نظام البطولة",                    ["48", "12", "104"]),
    ("static", "ar", "كم دور في البطولة",               ["6"]),
    ("static", "en", "how many world cups does Germany have", ["4", "1954"]),
    ("static", "ar", "هل ميسي بيلعب",                   ["38"]),

    # rag (in-corpus 2026 facts)
    ("rag",    "ar", "متى تبدأ بطولة كأس العالم 2026؟", ["11 يونيو", "يونيو 2026"]),
    ("rag",    "ar", "وين النهائي",                     ["ميتلايف"]),
    ("rag",    "ar", "كم عدد الملاعب",                  ["16"]),
    ("rag",    "ar", "مين في مجموعة السعودية",          ["إسبانيا", "أوروغواي"]),
    ("rag",    "ar", "احتمال فوز المكسيك على جنوب أفريقيا", ["70.9", "70"]),
    ("rag",    "en", "Where is the final played?",       ["MetLife"]),
    ("rag",    "en", "How many host cities in the USA?", ["11"]),

    # out-of-scope (must refuse politely)
    ("oos",    "ar", "كم سعر التذكرة",                  ["تذاكر", "fifa", "ما عندي"]),
    ("oos",    "ar", "ب",                               ["ما عندي", "جرب"]),
    ("oos",    "ar", "كيف اطبخ مكرونة",                 ["ما عندي", "جرب"]),
]


def _one_stream(message: str, language: str, session_id: str
                ) -> tuple[float, float, str, list[dict[str, Any]]]:
    """Send one /chat/stream request; return (ttft, total, answer, sources)."""
    body = json.dumps(
        {"message": message, "language": language, "top_k": 5,
         "session_id": session_id},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    t0 = time.time()
    ttft: float | None = None
    answer = ""
    sources: list[dict[str, Any]] = []

    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        buf = b""
        for chunk in resp:
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                if not frame.startswith(b"data:"):
                    continue
                payload = frame[5:].strip()
                if not payload:
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if evt["type"] == "meta":
                    sources = evt.get("sources", [])
                elif evt["type"] == "token":
                    if ttft is None:
                        ttft = time.time() - t0
                    answer += evt.get("text", "")
                elif evt["type"] == "replace":
                    answer = evt.get("text", "")
                elif evt["type"] == "done":
                    break
                elif evt["type"] == "error":
                    break
    return ttft or (time.time() - t0), time.time() - t0, answer.strip(), sources


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main() -> None:
    session_id = str(uuid.uuid4())  # fresh session so no cross-pollution
    rows: list[dict[str, Any]] = []

    for category, language, message, expect in QUERIES:
        ttft, total, answer, sources = _one_stream(message, language, session_id)
        norm = answer.lower()
        hit = any(kw.lower() in norm for kw in expect)
        # Heuristic: a non-LLM route returns the whole answer in the first
        # token, so ttft ~= total.
        is_fast_path = (total - ttft) < 0.05
        rows.append({
            "category": category,
            "language": language,
            "question": message,
            "answer": answer,
            "expect_any": expect,
            "hit": hit,
            "ttft_s": round(ttft, 3),
            "total_s": round(total, 3),
            "sources": len(sources),
            "route": "fast" if is_fast_path else "llm",
        })

    # ---- summary ----
    cats: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)

    def _stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        totals = [r["total_s"] for r in items]
        ttfts = [r["ttft_s"] for r in items]
        hits = [r["hit"] for r in items]
        return {
            "n": len(items),
            "hit_rate": round(sum(hits) / len(hits), 3) if hits else 0.0,
            "total_p50": round(_pct(totals, 50), 3),
            "total_p95": round(_pct(totals, 95), 3),
            "total_mean": round(statistics.mean(totals), 3) if totals else 0.0,
            "ttft_p50": round(_pct(ttfts, 50), 3),
            "ttft_p95": round(_pct(ttfts, 95), 3),
        }

    report = {
        "session_id": session_id,
        "n_queries": len(rows),
        "summary_overall": _stats(rows),
        "summary_by_category": {c: _stats(items) for c, items in cats.items()},
        "queries": rows,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_FILE}  —  {len(rows)} queries")
    print(f"overall: {report['summary_overall']}")
    for cat, stats in report["summary_by_category"].items():
        print(f"  {cat:8s}: {stats}")


if __name__ == "__main__":
    main()
