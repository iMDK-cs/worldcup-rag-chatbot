"""Generate large-scale synthetic Q&A pairs for World Cup 2026 fine-tuning.

Produces an instruction-tuning dataset (``{instruction, input, output, source}``)
sized for a real fine-tune (default 5000 pairs) across three streams:

    * ``ar_msa``    — Modern Standard Arabic (50%)
    * ``ar_saudi``  — Saudi / Khaleeji colloquial (20%)
    * ``en``        — English (30%)

Each batch is grounded in the actual fact corpus (host-cities CSV, schedule
CSV, probabilities CSV, manual Q&A) so the model can't invent dates or
stadiums. Output is streamed to JSONL with dedup + validation so a Ctrl-C
loses at most one batch.

Usage:
    # Generate the full 5000-pair set (resumable)
    uv run python -m src.training.qa_generator_v2

    # Small sanity batch
    uv run python -m src.training.qa_generator_v2 --target 100

    # Switch model
    uv run python -m src.training.qa_generator_v2 --model claude-opus-4-7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ValidationError

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

Stream = Literal["ar_msa", "ar_saudi", "en"]

DEFAULT_TARGET: int = 5000
DEFAULT_MODEL: str = "claude-sonnet-4-6"
PAIRS_PER_BATCH: int = 50
MAX_RETRIES: int = 3
RETRY_BACKOFF_S: float = 4.0

STREAM_SHARES: dict[Stream, float] = {
    "ar_msa": 0.50,
    "ar_saudi": 0.20,
    "en": 0.30,
}

TOPICS: tuple[str, ...] = (
    "dates_and_schedule",
    "venues_and_cities",
    "groups_and_teams",
    "results_and_probabilities",
    "history_and_records",
    "saudi_arabia_focus",
    "general_tournament_facts",
)
TOPIC_WEIGHTS: tuple[float, ...] = (0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05)

OUTPUT_FILE: Path = settings.paths.synthetic / "qa_v2_dataset.jsonl"
PROGRESS_FILE: Path = settings.paths.synthetic / "qa_v2_progress.json"

BANNED_PREFIXES: tuple[str, ...] = (
    "بناءً على",
    "بناء على",
    "وفقاً لـ",
    "وفقا لـ",
    "Based on",
    "According to",
    "As an AI",
    "I'm sorry",
)


# ---------------------------------------------------------------------------
# Fact corpus (compact reference the model sees in every prompt)
# ---------------------------------------------------------------------------

CORE_FACTS = """\
GENERAL
- 2026 FIFA World Cup: 48 teams, 12 groups (A–L), 104 matches total.
- Group stage: 72 matches (12 groups × 6 each). Knockouts: 32 matches.
- Hosts: United States (11 cities), Canada (2), Mexico (3). Total 16 host cities.
- Tournament opens: 11 June 2026 at Estadio Azteca, Mexico City.
- Tournament final: 19 July 2026 at MetLife Stadium, New York/New Jersey.
- MetLife Stadium hosts the most matches, including the final.

GROUPS (selected)
- Group A: Mexico (host), South Africa, South Korea, UEFA Playoff D.
- Group C: Morocco, Brazil, Haiti, Scotland.
- Group H: Saudi Arabia, Spain, Uruguay, Cape Verde.

ARAB TEAMS QUALIFIED
- Saudi Arabia, Qatar, Morocco, Tunisia, Egypt, Algeria, Jordan, Iraq (8 nations).

MEXICAN HOST CITIES
- Mexico City — Estadio Azteca
- Guadalajara — Estadio Akron
- Monterrey — Estadio BBVA

CANADIAN HOST CITIES
- Toronto — BMO Field
- Vancouver — BC Place

SAMPLE PROBABILITIES (Group A baseline model)
- Mexico vs South Africa: Mexico 70.9% / Draw 15.8% / South Africa 13.4%.
- Mexico vs South Korea: Mexico 54.7% / Draw 20.1% / South Korea 25.1%.

KEY DATES
- 11 June 2026 — Opening match (Mexico City).
- 13 June 2026 — Saudi Arabia's first match.
- 19 July 2026 — Final at MetLife Stadium.
"""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    instruction: str
    input: str = ""
    output: str
    source: str = "synthetic_v2"


# ---------------------------------------------------------------------------
# Stream-specific system prompts
# ---------------------------------------------------------------------------

_SYS_HEADER_AR = (
    "أنت مساعد ذكي وكاتب رياضي محترف، تنشئ بيانات تدريب لنموذج لغوي عن "
    "كأس العالم 2026. كل زوج لازم يكون مأخوذ مباشرة من «حقائق المرجع» "
    "اللي بنعطيك إياها. ممنوع تخترع أي معلومة جديدة."
)

_SYS_HEADER_EN = (
    "You are a sports writer creating instruction-tuning data for a 2026 "
    "FIFA World Cup chatbot. Every Q&A pair MUST be grounded in the "
    "reference facts provided. Do not invent dates, scores, or names."
)

_SYS_MSA = f"""{_SYS_HEADER_AR}

قواعد لهذه الدفعة:
- استخدم الفصحى الحديثة فقط (مثل أسلوب التعليق الرياضي العربي على بي ان سبورتس).
- صياغة طبيعية، ليست روبوتية.
- لا تبدأ بـ «بناءً على» أو «وفقاً لـ».
- جواب من 1 إلى 3 جمل قصيرة.
- إذا السؤال عن السعودية أضف لمسة حماس خفيفة (مرة وحدة كحد أقصى).
- نوّع الأسئلة بين «متى/كم/ما هي/أين/من».

حقائق المرجع:
{CORE_FACTS}"""

_SYS_SAUDI = f"""{_SYS_HEADER_AR}

قواعد لهذه الدفعة (لهجة سعودية/خليجية):
- استخدم اللهجة السعودية الطبيعية كأنك تتكلم مع صاحبك في الستاد.
- الكلمات: «وش» بدل «ماذا»، «مين» بدل «من»، «وين» بدل «أين»،
  «ليش» بدل «لماذا»، «كم» للأعداد، «متى» للوقت، «الحين» للوقت الحاضر،
  «أبشر»، «يا بطل»، «صراحة»، «والله».
- ممنوع الفصحى مثل «أين/ماذا/لماذا/كيف/هل».
- الأخطاء الإملائية البسيطة في الأسئلة مقبولة (محاكاة المستخدم الحقيقي):
  مثلاً «السعوديه» بدون همزة، «ليش» بدل «لماذا».
- الجواب 1-3 جمل، نبرة سوالف.
- لا تبدأ بـ «بناءً على» أو «وفقاً لـ».
- إذا السؤال عن السعودية، حماس عادي مثل «وعد منا!» أو «إن شاء الله نفرح».

أمثلة على اللهجة المطلوبة:
س: «وين بيلعبون النهائي يا بطل؟»
ج: «النهائي الكبير راح يكون في ملعب ميتلايف بنيويورك يوم 19 يوليو 2026، الوعد هناك!»

س: «متى تلعب السعوديه أول مباراه؟»
ج: «السعودية تبدأ مشوارها يوم 13 يونيو 2026، إن شاء الله بداية موفقة.»

حقائق المرجع:
{CORE_FACTS}"""

_SYS_EN = f"""{_SYS_HEADER_EN}

Rules for this batch (English):
- Conversational, like a sports columnist — not stiff.
- Never start with "Based on" or "According to".
- Answer in 1–3 short sentences.
- Vary question stems: "When / Where / How many / Which / Who / What".
- Use clear specifics from the reference (dates, stadiums, cities).

Reference facts:
{CORE_FACTS}"""

SYS_BY_STREAM: dict[Stream, str] = {
    "ar_msa": _SYS_MSA,
    "ar_saudi": _SYS_SAUDI,
    "en": _SYS_EN,
}


# ---------------------------------------------------------------------------
# Fact pool — drawn from CSVs each batch to seed varied questions
# ---------------------------------------------------------------------------

def _load_fact_seeds() -> list[str]:
    """Pull structured facts from CSVs to seed varied questions per batch."""
    seeds: list[str] = []

    sched = settings.paths.csvs / "FIFA2026_schedule.csv"
    if sched.exists():
        with sched.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seeds.append(
                    f"{r['match_number']} of Group {r['group']} on "
                    f"{r['date']} at {r['stadium']}."
                )

    cities = settings.paths.csvs / "host_cities.csv"
    if cities.exists():
        with cities.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seeds.append(
                    f"{r['city_name']}, {r['country']} hosts at {r['venue_name']} "
                    f"(airport {r['airport_code']})."
                )

    probs = settings.paths.csvs / "future_match_probabilities_baseline.csv"
    if probs.exists():
        with probs.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    p_h = float(r["p_home_win"])
                    p_d = float(r["p_draw"])
                    p_a = float(r["p_away_win"])
                except (ValueError, KeyError):
                    continue
                seeds.append(
                    f"Group {r['group']}: {r['home_team']} vs {r['away_team']} "
                    f"— win {p_h:.1%}, draw {p_d:.1%}, lose {p_a:.1%}."
                )

    for fname in ("qa_manual_en.jsonl", "qa_manual_ar.jsonl"):
        p = settings.paths.synthetic / fname
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    seeds.append(f"{rec.get('question','')} → {rec.get('answer','')}")
                except json.JSONDecodeError:
                    pass

    random.shuffle(seeds)
    return seeds


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_user_prompt(
    stream: Stream, batch_size: int, topic: str, seeds: list[str]
) -> str:
    """Build the per-batch user message with topic + seed facts."""
    seed_block = "\n".join(f"- {s}" for s in seeds[:15])

    lang_instruction = {
        "ar_msa": "اكتب الأسئلة والأجوبة بالعربية الفصحى الحديثة.",
        "ar_saudi": "اكتب الأسئلة والأجوبة باللهجة السعودية / الخليجية.",
        "en": "Write the questions and answers in natural English.",
    }[stream]

    return f"""Generate exactly {batch_size} unique instruction/output pairs focused on the topic **{topic}**.

{lang_instruction}

Vary phrasing, vary question stems, and cover different sub-aspects of the topic. Each pair MUST be grounded in either the core reference (in the system prompt) or one of these structured seeds:

{seed_block}

Return ONLY a JSON array of {batch_size} objects, each shaped:
{{"instruction": "...", "output": "...", "topic": "{topic}"}}

The `input` field is omitted (always empty for this dataset). Do not include any prose outside the JSON array. Do not number the items."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid(pair: dict[str, Any], stream: Stream) -> tuple[bool, str]:
    """Cheap rule-based validation. Returns (ok, reason_if_bad)."""
    if not isinstance(pair, dict):
        return False, "not a dict"

    instr = pair.get("instruction", "")
    out = pair.get("output", "")

    if not isinstance(instr, str) or not isinstance(out, str):
        return False, "non-string instr/output"

    instr = instr.strip()
    out = out.strip()

    if not instr or not out:
        return False, "empty"

    if len(instr) < 5 or len(instr) > 400:
        return False, f"instruction length {len(instr)}"
    if len(out) < 5 or len(out) > 800:
        return False, f"output length {len(out)}"

    for bad in BANNED_PREFIXES:
        if out.lower().startswith(bad.lower()):
            return False, f"banned prefix: {bad!r}"

    if stream == "ar_saudi":
        # The question is where dialect shows most reliably; the answer can be
        # more neutral and still feel natural in Saudi/Khaleeji speech.
        dialect_markers = ("وش", "مين", "وين", "ليش", "كيف", "ايش", "إيش",
                           "متى", "كم", "الحين", "هذي", "هذا", "ذي")
        if not any(m in instr for m in dialect_markers):
            return False, "missing Saudi dialect marker in question"

    return True, ""


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------

def _call_claude(
    client: anthropic.Anthropic, model: str, system: str, user: str
) -> list[dict[str, Any]]:
    """One API call → parsed list of pair dicts. Retries on transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                b.text for b in msg.content if getattr(b, "type", "") == "text"
            )
            return _parse_json_array(text)
        except (anthropic.APIError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
            else:
                raise
    return []


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Pull the first JSON array out of the model's response."""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in response")

    arr = json.loads(text[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("top-level JSON is not an array")
    return arr


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _existing_counts() -> dict[Stream, int]:
    """Tally how many pairs we already have per stream (for resume)."""
    counts: Counter[Stream] = Counter()
    if not OUTPUT_FILE.exists():
        return dict(counts)
    for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stream = rec.get("stream")
        if stream in STREAM_SHARES:
            counts[stream] += 1  # type: ignore[index]
    return dict(counts)


def _existing_instructions() -> set[str]:
    """Set of instruction strings already seen (for dedup)."""
    seen: set[str] = set()
    if not OUTPUT_FILE.exists():
        return seen
    for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            instr = rec.get("instruction", "").strip().lower()
            if instr:
                seen.add(instr)
        except json.JSONDecodeError:
            pass
    return seen


def _pick_topic() -> str:
    """Sample a topic according to TOPIC_WEIGHTS."""
    return random.choices(TOPICS, weights=TOPIC_WEIGHTS, k=1)[0]


def run(target: int, model: str) -> None:
    """Main loop: top up each stream to its target share, in batches."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    targets: dict[Stream, int] = {
        s: int(round(target * share)) for s, share in STREAM_SHARES.items()
    }
    # Fix rounding drift so the targets sum exactly to ``target``.
    drift = target - sum(targets.values())
    targets["ar_msa"] += drift

    counts = _existing_counts()
    seen_instructions = _existing_instructions()
    seeds = _load_fact_seeds()
    logger.info(
        "Targets per stream: %s | already have: %s | seeds loaded: %d",
        targets, counts, len(seeds),
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Per-stream stats for the run summary.
    rejected = defaultdict(int)
    written = defaultdict(int)

    while True:
        remaining: dict[Stream, int] = {
            s: targets[s] - counts.get(s, 0) for s in STREAM_SHARES
        }
        if all(r <= 0 for r in remaining.values()):
            break

        # Pick the stream furthest from its target proportion.
        stream: Stream = max(remaining, key=remaining.get)  # type: ignore[arg-type]
        batch_size = min(PAIRS_PER_BATCH, remaining[stream])
        topic = _pick_topic()

        logger.info(
            "Generating %d pairs | stream=%s | topic=%s | remaining: %s",
            batch_size, stream, topic, remaining,
        )

        random.shuffle(seeds)
        user_prompt = _build_user_prompt(stream, batch_size, topic, seeds)
        try:
            raw_pairs = _call_claude(client, model, SYS_BY_STREAM[stream], user_prompt)
        except Exception:
            logger.exception("Batch failed permanently — skipping.")
            continue

        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            for p in raw_pairs:
                ok, why = _is_valid(p, stream)
                if not ok:
                    rejected[stream] += 1
                    continue

                instr = p["instruction"].strip()
                if instr.lower() in seen_instructions:
                    rejected[stream] += 1
                    continue
                seen_instructions.add(instr.lower())

                rec = {
                    "instruction": instr,
                    "input": p.get("input", ""),
                    "output": p["output"].strip(),
                    "stream": stream,
                    "topic": p.get("topic", topic),
                    "source": "synthetic_v2",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written[stream] += 1
                counts[stream] = counts.get(stream, 0) + 1

        logger.info(
            "Batch done — written this batch: %d kept / %d rejected",
            sum(written.values()) - sum(written.get(s, 0) for s in STREAM_SHARES if s != stream),
            rejected.get(stream, 0),
        )
        PROGRESS_FILE.write_text(
            json.dumps({"counts": counts, "targets": targets,
                        "rejected": dict(rejected), "model": model}, indent=2),
            encoding="utf-8",
        )

    logger.info("DONE — final counts: %s | rejected: %s", counts, dict(rejected))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate large-scale World Cup 2026 Q&A pairs via the Anthropic API."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Total pairs to generate (default {DEFAULT_TARGET}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Anthropic model id (default {DEFAULT_MODEL!r}).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for topic / seed-fact sampling.")
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    random.seed(args.seed)

    if not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set. Add it to .env.")
        sys.exit(1)

    run(target=args.target, model=args.model)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _main()
