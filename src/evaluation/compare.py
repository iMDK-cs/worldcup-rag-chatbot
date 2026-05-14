"""Compare RAG-only vs RAG + fine-tuned Mistral on a 10-question bilingual set.

Architecture (works around a Windows + bitsandbytes + PEFT segfault that
fires when ``PeftModel.from_pretrained`` is called on a live 4-bit model):

    1. The ``rag_only`` subprocess loads the 4-bit base alone and writes
       ``models/eval_partial_rag_only.json``.
    2. The ``rag_finetuned`` subprocess loads the 4-bit base + adapter via
       ``AutoPeftModelForCausalLM`` (single shot — no live attach) and writes
       ``models/eval_partial_rag_finetuned.json``.
    3. The default ``--mode all`` orchestrator runs both subprocesses, then
       merges them into ``models/eval_results.json``.

Usage:
    uv run python -m src.evaluation.compare              # full pipeline
    uv run python -m src.evaluation.compare --mode rag_only
    uv run python -m src.evaluation.compare --mode rag_finetuned
    uv run python -m src.evaluation.compare --mode merge
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.api.chat import (
    BASE_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    _build_messages,
    _format_context,
)
from src.config import settings

logger = logging.getLogger(__name__)


EVAL_QUESTIONS: list[dict[str, str]] = [
    # English
    {"language": "en", "question": "Who are the host countries of the FIFA World Cup 2026?"},
    {"language": "en", "question": "Which cities in Mexico are hosting 2026 World Cup matches?"},
    {"language": "en", "question": "What is the probability of Mexico beating South Africa in Group A?"},
    {"language": "en", "question": "At which stadium will Match 1 of the 2026 World Cup be played?"},
    {"language": "en", "question": "When does the 2026 FIFA World Cup tournament start?"},
    # Arabic
    {"language": "ar", "question": "وش مدن المكسيك اللي تستضيف كأس العالم 2026؟"},
    {"language": "ar", "question": "متى تبدأ بطولة كأس العالم 2026؟"},
    {"language": "ar", "question": "ما هي احتمالية فوز المكسيك على جنوب أفريقيا في المجموعة A؟"},
    {"language": "ar", "question": "وين تنلعب مباراة 1 من كأس العالم 2026؟"},
    {"language": "ar", "question": "كم مدينة تستضيف مباريات كأس العالم 2026؟"},
]

ADAPTER_DIR: Path = settings.paths.models / "mistral-7b-worldcup"
OUTPUT_JSON: Path = settings.paths.models / "eval_results.json"
HITS_JSON: Path = settings.paths.models / "eval_hits.json"
PARTIAL_RAG_ONLY: Path = settings.paths.models / "eval_partial_rag_only.json"
PARTIAL_RAG_TUNED: Path = settings.paths.models / "eval_partial_rag_finetuned.json"
TOP_K: int = 5


# ---------------------------------------------------------------------------
# Model loaders (each runs in its own subprocess to keep the GPU clean)
# ---------------------------------------------------------------------------

def _load_base_only() -> tuple[Any, Any]:
    """Load the 4-bit base, no adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, token=settings.hf_token, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Loading base model %s …", BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        token=settings.hf_token,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = True
    model.eval()
    return tokenizer, model


def _load_with_adapter() -> tuple[Any, Any]:
    """Load 4-bit base + LoRA adapter in one shot via AutoPeftModelForCausalLM."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR), use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Loading PEFT model from %s …", ADAPTER_DIR)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER_DIR),
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        token=settings.hf_token,
    )
    model.config.use_cache = True
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Greedy-ish generation; returns only the new assistant tokens."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _precompute_hits() -> None:
    """Run retrieval for every eval question and dump to disk.

    Done in a dedicated step *before* any LLM is loaded so the embedder /
    ChromaDB native libs never coexist with bitsandbytes / Mistral in the
    same process (that combination segfaults on Windows).
    """
    from src.rag.pipeline import retrieve  # lazy: only when we actually need it

    rows: list[dict[str, Any]] = []
    for q in EVAL_QUESTIONS:
        hits = retrieve(q["question"], top_k=TOP_K)
        rows.append(
            {
                "question": q["question"],
                "language": q["language"],
                "sources": [
                    {"text": h.text, "source": h.source, "score": h.score, "metadata": h.metadata}
                    for h in hits
                ],
            }
        )

    HITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    HITS_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d retrieval rows to %s", len(rows), HITS_JSON)


def _run_one_mode(mode: str, out_path: Path) -> None:
    """Generate one column (rag_only or rag_finetuned) using pre-computed hits."""
    if not HITS_JSON.exists():
        raise FileNotFoundError(
            f"Pre-computed retrieval missing at {HITS_JSON}. Run the "
            f"orchestrator (--mode all) which produces it first."
        )
    hits_rows = json.loads(HITS_JSON.read_text(encoding="utf-8"))

    if mode == "rag_only":
        tokenizer, model = _load_base_only()
    elif mode == "rag_finetuned":
        tokenizer, model = _load_with_adapter()
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(hits_rows, 1):
        # Re-wrap the saved hit dicts into the format _format_context expects.
        from src.rag.pipeline import RetrievalHit

        hits = [
            RetrievalHit(
                text=s["text"],
                source=s["source"],
                score=s["score"],
                metadata=s.get("metadata", {}),
            )
            for s in row["sources"]
        ]
        context_block = _format_context(hits)
        messages = _build_messages(row["question"], row["language"], context_block)

        answer = _generate(model, tokenizer, messages)

        rows.append(
            {
                "question": row["question"],
                "language": row["language"],
                "sources": row["sources"],
                "answer": answer,
            }
        )
        logger.info("[%s] %d/%d  %s — %s", mode, i, len(hits_rows), row["language"], row["question"][:60])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d rows to %s", len(rows), out_path)


def _merge() -> None:
    """Combine the two partial JSONs into the final eval_results.json."""
    if not PARTIAL_RAG_ONLY.exists() or not PARTIAL_RAG_TUNED.exists():
        raise FileNotFoundError(
            f"Both partials must exist before merge:\n"
            f"  {PARTIAL_RAG_ONLY} — {'OK' if PARTIAL_RAG_ONLY.exists() else 'MISSING'}\n"
            f"  {PARTIAL_RAG_TUNED} — {'OK' if PARTIAL_RAG_TUNED.exists() else 'MISSING'}"
        )

    rag_only = json.loads(PARTIAL_RAG_ONLY.read_text(encoding="utf-8"))
    rag_tuned = json.loads(PARTIAL_RAG_TUNED.read_text(encoding="utf-8"))

    if len(rag_only) != len(rag_tuned):
        raise RuntimeError(
            f"Partial row counts differ: rag_only={len(rag_only)} vs "
            f"rag_finetuned={len(rag_tuned)}"
        )

    merged: list[dict[str, Any]] = []
    for a, b in zip(rag_only, rag_tuned):
        if a["question"] != b["question"]:
            raise RuntimeError(
                f"Question mismatch between partials: {a['question']!r} != {b['question']!r}"
            )
        merged.append(
            {
                "question": a["question"],
                "language": a["language"],
                "sources": a["sources"],
                "rag_only": a["answer"],
                "rag_finetuned": b["answer"],
            }
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote merged eval (%d rows) to %s", len(merged), OUTPUT_JSON)

    for r in merged:
        print("\n" + "=" * 80)
        print(f"[{r['language']}] Q: {r['question']}")
        print(f"  RAG-only       : {r['rag_only'][:300]}")
        print(f"  RAG+finetuned  : {r['rag_finetuned'][:300]}")


def _run_subprocess(mode: str) -> None:
    """Spawn ``python -m src.evaluation.compare --mode <mode>``."""
    logger.info("Spawning subprocess for mode=%s", mode)
    result = subprocess.run(
        [sys.executable, "-u", "-m", "src.evaluation.compare", "--mode", mode],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess for mode={mode!r} exited with code {result.returncode}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bilingual RAG eval comparison.")
    parser.add_argument(
        "--mode",
        choices=("all", "retrieve", "rag_only", "rag_finetuned", "merge"),
        default="all",
    )
    args = parser.parse_args(argv)

    if args.mode == "retrieve":
        _precompute_hits()
    elif args.mode == "rag_only":
        _run_one_mode("rag_only", PARTIAL_RAG_ONLY)
    elif args.mode == "rag_finetuned":
        _run_one_mode("rag_finetuned", PARTIAL_RAG_TUNED)
    elif args.mode == "merge":
        _merge()
    else:  # "all"
        _run_subprocess("retrieve")
        _run_subprocess("rag_only")
        _run_subprocess("rag_finetuned")
        _merge()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
