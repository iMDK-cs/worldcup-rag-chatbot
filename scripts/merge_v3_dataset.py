"""دمج qa_v2_dataset.jsonl + qa_gemini.jsonl + qa_claude.jsonl → qa_v3_dataset.jsonl

- يطبّع schema الأسطر الجديدة (يضيف stream/topic/source).
- يكتشف اللغة بشكل بسيط (حروف لاتينية vs عربية).
- يحذف المكرر بمطابقة instruction (case-insensitive, مع تطبيع المسافات).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA = Path("data/synthetic")
INPUTS = [
    (DATA / "qa_v2_dataset.jsonl", "synthetic_v2"),
    (DATA / "qa_gemini.jsonl", "gemini_v3"),
    (DATA / "qa_claude.jsonl", "claude_v3"),
]
OUTPUT = DATA / "qa_v3_dataset.jsonl"

LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
WS_RE = re.compile(r"\s+")


def detect_stream(text: str) -> str:
    latin = len(LATIN_RE.findall(text))
    arabic = len(ARABIC_RE.findall(text))
    if latin > arabic:
        return "en"
    return "ar_msa"  # لا نقدر نميّز فصحى/عامية بدقة هنا


def normalize_key(s: str) -> str:
    return WS_RE.sub(" ", s.strip().lower())


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                # أسطر فيها JSON ناقص - تجاهل بهدوء
                continue


def main() -> int:
    seen: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    out_records: list[dict] = []

    for path, source_tag in INPUTS:
        if not path.exists():
            print(f"⚠️  ملف مفقود: {path}", file=sys.stderr)
            continue

        loaded = kept = dup = bad = 0
        for rec in iter_jsonl(path):
            loaded += 1
            instr = rec.get("instruction")
            output = rec.get("output")
            if not instr or not output:
                bad += 1
                continue

            key = normalize_key(instr)
            if key in seen:
                dup += 1
                continue
            seen.add(key)

            merged = {
                "instruction": instr.strip(),
                "input": rec.get("input", ""),
                "output": output.strip(),
                "stream": rec.get("stream") or detect_stream(instr),
                "topic": rec.get("topic", "mixed"),
                "source": rec.get("source", source_tag),
            }
            out_records.append(merged)
            kept += 1

        stats[path.name] = {"loaded": loaded, "kept": kept, "dup": dup, "bad": bad}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"📊 تقرير الدمج → {OUTPUT}")
    print("=" * 60)
    for name, s in stats.items():
        print(
            f"  {name:30s}  loaded={s['loaded']:>5}  kept={s['kept']:>5}  "
            f"dup={s['dup']:>4}  bad={s['bad']:>4}"
        )
    print("-" * 60)
    print(f"✅ الإجمالي النهائي: {len(out_records)} زوج Q&A فريد")

    # توزيع حسب stream/source
    by_stream: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in out_records:
        by_stream[r["stream"]] = by_stream.get(r["stream"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"   by stream:  {by_stream}")
    print(f"   by source:  {by_source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
