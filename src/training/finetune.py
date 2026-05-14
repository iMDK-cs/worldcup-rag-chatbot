"""Local QLoRA fine-tuning of Mistral-7B on the synthetic World Cup Q&A.

Pure HuggingFace + PEFT + TRL + bitsandbytes pipeline (no Unsloth) so it
runs cleanly on Windows with CUDA 12.4 / RTX 3080-class GPUs.

Outputs (under ``settings.paths.models``):
    * ``mistral-7b-worldcup/``     — LoRA adapter + tokenizer
    * ``training_loss.png``        — per-step loss plot
    * ``eval_before_after.json``   — baseline vs tuned generations

Usage:
    uv run python -m src.training.finetune          # full pipeline
    uv run python -m src.training.finetune --no-eval  # train + save only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_SUBDIR: str = "mistral-7b-worldcup"
LOSS_PLOT_NAME: str = "training_loss.png"
EVAL_JSON_NAME: str = "eval_before_after.json"

MAX_SEQ_LENGTH: int = 1024
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
PER_DEVICE_BS: int = 2
GRAD_ACCUM: int = 8
NUM_EPOCHS: int = 2
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 0.01
MAX_GRAD_NORM: float = 1.0
SEED: int = 42

EVAL_SPLIT_FRAC: float = 0.05  # 5% held out for overfit detection
EVAL_STEPS: int = 50
SAVE_STEPS: int = 50
SAVE_TOTAL_LIMIT: int = 2

EVAL_QUESTIONS: list[str] = [
    "Who is in Group A of FIFA World Cup 2026?",
    "ما هي مجموعة السعودية في كأس العالم 2026؟",
    "What is the probability of Mexico winning against South Africa?",
    "When does Morocco play Brazil in Group C?",
    "ما هي احتمالية فوز فرنسا على السنغال؟",
]

LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_qa_records(qa_dir: Path) -> list[dict[str, str]]:
    """Load training Q&A records.

    Prefers the merged ``qa_train_all.jsonl`` (instruction/output schema,
    produced by :mod:`src.training.merge_datasets`). Falls back to the
    older ``qa_arabic_all.jsonl`` + ``qa_english_all.jsonl`` pair (which
    use the question/answer schema) so legacy data still works.

    Each returned record is normalised to
    ``{"question": str, "answer": str, "language": "ar"|"en"}``.
    """
    # Prefer the cleaned + gap-filled file when present.
    clean = qa_dir / "qa_train_clean.jsonl"
    merged = clean if clean.exists() else (qa_dir / "qa_train_all.jsonl")
    if merged.exists():
        records: list[dict[str, str]] = []
        with merged.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(
                    {
                        "question": rec["instruction"],
                        "answer": rec["output"],
                        "language": rec.get("language", "en"),
                    }
                )
        if records:
            logger.info("Loaded %d records from %s (merged)", len(records), merged)
            return records

    records = []
    for name in ("qa_arabic_all.jsonl", "qa_english_all.jsonl"):
        path = qa_dir / name
        if not path.exists():
            logger.warning("Missing %s — skipping.", path)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        raise FileNotFoundError(
            f"No Q&A JSONL files found under {qa_dir}. "
            f"Run `python -m src.training.merge_datasets` first."
        )
    return records


def _build_hf_dataset(records: list[dict[str, str]], tokenizer: Any) -> Any:
    """Render Q&A records through Mistral's chat template into a single ``text`` column."""
    from datasets import Dataset

    texts: list[str] = []
    for r in records:
        messages = [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return Dataset.from_dict({"text": texts})


def _split_records(
    records: list[dict[str, str]], frac: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Stratified train/eval split by language so AR/EN ratios match in both."""
    import random

    rng = random.Random(seed)
    by_lang: dict[str, list[dict[str, str]]] = {}
    for r in records:
        by_lang.setdefault(r.get("language", "en"), []).append(r)

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for lang, items in by_lang.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * frac)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
        logger.info("  lang=%s  train=%d  val=%d", lang, len(items) - n_val, n_val)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_tokenizer() -> Any:
    """Load the Mistral tokenizer with a usable pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=settings.hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_base_model() -> Any:
    """Load Mistral-7B in 4-bit NF4 with bitsandbytes."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=settings.hf_token,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    return model


def _attach_lora(model: Any) -> Any:
    """Wrap the 4-bit base with PEFT LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    work_dir: Path,
) -> Any:
    """Run SFT with TRL's SFTTrainer, with eval-driven best-checkpoint selection."""
    from trl import SFTConfig, SFTTrainer

    work_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(work_dir),
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=config,
    )
    trainer.train()
    return trainer


def _plot_loss(trainer: Any, out_path: Path) -> None:
    """Save a PNG of train + eval loss curves to spot over/under-fitting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []

    for entry in trainer.state.log_history:
        step = int(entry["step"]) if "step" in entry else None
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(float(entry["eval_loss"]))

    if not train_losses and not eval_losses:
        logger.warning("No loss recorded — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, marker="o", linewidth=1.5,
                label="train", color="#2563eb")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, marker="s", linewidth=1.8,
                label="eval (val 5%)", color="#dc2626")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Mistral-7B QLoRA — World Cup 2026 Q&A (6,945 pairs)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved loss plot to %s", out_path)


# ---------------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------------

def _generate(model: Any, tokenizer: Any, question: str, max_new_tokens: int = 256) -> str:
    """Generate one answer with deterministic decoding."""
    import torch

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        if was_training:
            model.train()

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _evaluate(
    baseline_outputs: dict[str, str],
    tuned_model: Any,
    tokenizer: Any,
    questions: list[str],
    out_json: Path,
) -> None:
    """Generate tuned-model answers and write the before/after JSON."""
    rows: list[dict[str, str]] = []
    for q in questions:
        after = _generate(tuned_model, tokenizer, q)
        rows.append({"question": q, "before": baseline_outputs.get(q, ""), "after": after})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved evaluation JSON to %s", out_json)

    for r in rows:
        print("\n" + "=" * 80)
        print("Q:", r["question"])
        print("BEFORE:", r["before"])
        print("AFTER: ", r["after"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(skip_eval: bool = False) -> None:
    """End-to-end pipeline: load data, train, save, evaluate."""
    import torch

    qa_dir = settings.paths.synthetic
    model_dir = settings.paths.models / OUTPUT_SUBDIR
    loss_png = settings.paths.models / LOSS_PLOT_NAME
    eval_json = settings.paths.models / EVAL_JSON_NAME

    records = _load_qa_records(qa_dir)
    logger.info("Loaded %d Q&A records from %s", len(records), qa_dir)

    tokenizer = _build_tokenizer()
    model = _build_base_model()

    baseline_outputs: dict[str, str] = {}
    if not skip_eval:
        logger.info("Capturing baseline outputs from un-tuned model …")
        for q in EVAL_QUESTIONS:
            baseline_outputs[q] = _generate(model, tokenizer, q)

    model = _attach_lora(model)

    logger.info("Splitting %d records (val frac %.2f) …", len(records), EVAL_SPLIT_FRAC)
    train_records, val_records = _split_records(records, EVAL_SPLIT_FRAC, SEED)
    logger.info("Final split: train=%d  val=%d", len(train_records), len(val_records))

    train_ds = _build_hf_dataset(train_records, tokenizer)
    val_ds = _build_hf_dataset(val_records, tokenizer)

    trainer = _train(
        model,
        tokenizer,
        train_ds,
        val_ds,
        work_dir=settings.paths.models / f"{OUTPUT_SUBDIR}-checkpoints",
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Saved adapter + tokenizer to %s", model_dir)

    _plot_loss(trainer, loss_png)

    if not skip_eval:
        _evaluate(baseline_outputs, trainer.model, tokenizer, EVAL_QUESTIONS, eval_json)

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line flags for the module entry point."""
    parser = argparse.ArgumentParser(
        description="Local QLoRA fine-tune of Mistral-7B on World Cup 2026 Q&A."
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the before/after evaluation phase (train + save only).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    run(skip_eval=args.no_eval)
