"""FastAPI chat backend: RAG-augmented inference on the QLoRA-tuned Mistral.

Endpoints:
    GET  /health  — liveness + which model is loaded.
    POST /chat    — `{"message": str, "language": "ar"|"en"}` → grounded answer.
    GET  /        — serves the static chat UI (`static/index.html`).

The model is loaded once at startup. If the fine-tuned LoRA adapter exists
at ``models/mistral-7b-worldcup``, it is applied on top of the 4-bit base;
otherwise the server falls back to the un-tuned base so the API is still
usable before fine-tuning completes.

Run with:
    uv run uvicorn src.api.chat:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings
from src.data import static_facts
from src.rag.numpy_retriever import RetrievalHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR: Path = settings.paths.models / "mistral-7b-worldcup"
STATIC_DIR: Path = settings.paths.project_root / "static"
TOP_K: int = 5
MAX_NEW_TOKENS: int = 180   # most answers are 1–3 sentences; cuts gen time ~40%
TEMPERATURE: float = 0.0    # pure greedy — fastest + most deterministic
MAX_CONTEXT_CHARS: int = 1600  # tighter context = shorter prompt = faster prefill
RETRIEVAL_SCORE_THRESHOLD: float = 0.55  # below this we fall through to static_facts

# Conversation memory.
MAX_SESSIONS: int = 500             # LRU eviction beyond this
MAX_TURNS_PER_SESSION: int = 6      # = 6 user msgs + 6 assistant msgs in context
MIN_QUESTION_CHARS: int = 3              # ignore single-letter "ت" inputs


SYSTEM_PROMPT_EN = (
    "You are Mundial Chatbot — a bilingual (Arabic/English) assistant for "
    "the FIFA World Cup 2026.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY facts from the provided context. Do NOT invent dates, "
    "scores, rules, players, countries, or stadiums.\n"
    "2. If the context does not contain the answer, reply briefly: "
    "\"I don't have specific information about that — try asking about "
    "schedules, groups, host cities, or match probabilities.\" Do NOT guess.\n"
    "3. Never start with phrases like 'Based on' or 'According to'.\n"
    "4. Keep answers short (1–3 sentences). Numbers and names must come "
    "from the context verbatim.\n"
    "5. Always reply in the same language as the user's question."
)

SYSTEM_PROMPT_AR = (
    "أنت Mundial Chatbot — مساعد ثنائي اللغة (عربي/إنجليزي) متخصّص في كأس "
    "العالم 2026.\n\n"
    "قواعد صارمة:\n"
    "1. استخدم فقط الحقائق الموجودة في السياق المقدَّم. لا تخترع تواريخ ولا "
    "نتائج ولا قوانين ولا لاعبين ولا دول ولا ملاعب.\n"
    "2. إذا السياق ما يحتوي الإجابة، رد بإيجاز: «ما عندي معلومة محددة عن "
    "هذا — جرّب تسأل عن الجدول أو المجموعات أو المدن المستضيفة أو احتمالات "
    "المباريات.» ولا تخمّن أبدًا.\n"
    "3. لا تبدأ بـ «بناءً على» أو «وفقًا لـ».\n"
    "4. خلّ الإجابة قصيرة (1-3 جمل). الأرقام والأسماء لازم تكون من السياق "
    "بالنص.\n"
    "5. أجب دائمًا بنفس لغة السؤال."
)

FALLBACK_EN = (
    "I don't have specific information about that. Try asking about the "
    "schedule, groups, host cities, or match probabilities for the 2026 "
    "World Cup."
)
FALLBACK_AR = (
    "ما عندي معلومة محددة عن هذا. جرّب تسأل عن الجدول أو المجموعات أو "
    "المدن المستضيفة أو احتمالات المباريات في كأس العالم 2026."
)

# Patterns we KNOW are hallucinated by the model (legacy garbage baked into
# the LoRA weights from the v3/gemini generation runs). If a generated
# answer matches any of these we replace it with the fallback rather than
# letting the user see fabricated rules.
import re as _re
HALLUCINATION_OUTPUT_PATTERNS: tuple[_re.Pattern[str], ...] = (
    # Made-up "law/rule of N seconds" in Arabic & English (including the
    # "قاعدة الـ 5 ثوانٍ" variant the model emits even after retraining).
    _re.compile(r"(?:قانون|قاعدة)\s+(?:الـ?\s*)?\d+\s+ثوان"),
    _re.compile(r"رميات\s+التماس"),  # fake jargon — there are no "throw-in 5s rules"
    _re.compile(r"\bcaptain\s+only\b", _re.IGNORECASE),
    _re.compile(r"قانون\s+(?:الـ?)?كابتن\s+فقط"),
    _re.compile(r"^\s*5\s*ثوان", _re.MULTILINE),
    _re.compile(r"rule of \d+ seconds", _re.IGNORECASE),
)


def _is_hallucination(answer: str) -> bool:
    """True if the generated answer contains a known fabricated pattern."""
    for pat in HALLUCINATION_OUTPUT_PATTERNS:
        if pat.search(answer):
            return True
    return False


# ---------------------------------------------------------------------------
# Social-intent layer: short-circuits greetings, self-id, and well-known
# player questions BEFORE the RAG+LLM pipeline. Without this, the model
# either refuses (boring) or hallucinates (player participation).
# ---------------------------------------------------------------------------

# Match across Arabic alef variants and strip simple diacritics.
_ARABIC_DIACRITICS = _re.compile(r"[ً-ْٰـ]")

def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = _re.sub(r"[!\?\.\،\؟]+", " ", t)
    t = _re.sub(r"\s+", " ", t)
    return t

SOCIAL_INTENTS: list[dict[str, Any]] = [
    {
        "name": "greeting_status",
        "keywords": ["كيف حالك", "كيف الحال", "كيفك", "اخبارك", "اخبار",
                     "how are you", "how r u", "hru"],
        "ar": "بخير والله! جاهز أجاوب على أي سؤال عن كأس العالم 2026 ⚽",
        "en": "Doing great! Ready to answer anything about the 2026 World Cup ⚽",
    },
    {
        "name": "self_id",
        # Token-based matching (see _match_social_intent) handles arbitrary
        # word order, so "انت مين" and "مين انت" both match the same intent.
        "keywords": ["مين انت", "من انت", "انت مين", "انت من", "من نت", "مين نت",
                     "تعرف نفسك", "عرف نفسك", "انت ايش", "ايش انت", "تعريف",
                     "who are you", "what are you", "introduce yourself", "u are who",
                     "اسمك", "اسمك ايش", "وش اسمك", "your name"],
        "ar": "أنا Mundial Chatbot، مساعدك الذكي لكل شي عن كأس العالم 2026!",
        "en": "I'm Mundial Chatbot — your AI assistant for everything about the 2026 World Cup!",
    },
    {
        "name": "who_am_i",
        "keywords": ["من انا", "من أنا", "مين انا", "انا مين", "انا من",
                     "who am i", "what is my name", "what am i"],
        "ar": "أنت المستخدم اللي يسأل! 😄 وأنا هنا أجاوب على أسئلتك عن كأس العالم 2026.",
        "en": "You're the user asking the questions! 😄 I'm here to answer anything about the 2026 World Cup.",
    },
    {
        "name": "ronaldo",
        "keywords": ["رونالدو", "كريستيانو", "ronaldo", "cristiano"],
        "ar": "رونالدو راح يكون عمره 41 سنة وقت البطولة — مشاركته للحين غير مؤكدة رسميًا.",
        "en": "Ronaldo will be 41 during the tournament — his participation isn't officially confirmed yet.",
    },
    {
        "name": "messi",
        "keywords": ["ميسي", "مسي", "ليونيل ميسي", "messi", "lionel messi"],
        "ar": "ميسي راح يكون عمره 38 سنة وقت البطولة — موضوع مشاركته للحين غير مؤكد.",
        "en": "Messi will be 38 during the tournament — his participation isn't confirmed yet.",
    },
    {
        "name": "hello",
        "keywords": ["السلام عليكم", "سلام عليكم", "اهلا", "أهلا", "اهلين",
                     "مرحبا", "مرحباً", "مرحبا بك", "هلا", "هلا والله", "هلابك",
                     "هاي", "هاااي",
                     "hi", "hello", "hey", "yo", "greetings", "salam", "salaam"],
        "ar": "وعليكم السلام! أنا Mundial Chatbot، اسألني أي شي عن كأس العالم 2026 ⚽",
        "en": "Hi! I'm Mundial Chatbot — ask me anything about the 2026 World Cup ⚽",
    },
    {
        "name": "thanks",
        "keywords": ["شكرا", "مشكور", "يعطيك العافيه", "يعطيك العافية", "تسلم",
                     "thank you", "thanks", "thx", "appreciate"],
        "ar": "العفو، أي وقت! 🙌",
        "en": "You're welcome, anytime! 🙌",
    },
    {
        "name": "bye",
        "keywords": ["مع السلامه", "في امان الله", "وداعا", "تصبح على خير",
                     "bye", "goodbye", "see you", "see ya"],
        "ar": "مع السلامة! أراك في الملعب يوم 11 يونيو 2026 ⚽",
        "en": "Goodbye! See you at the opening match on 11 June 2026 ⚽",
    },
    {
        "name": "most_wins",
        "keywords": ["اكثر منتخب فاز", "أكثر منتخب فاز", "اكثر منتخب حقق",
                     "most world cup wins", "most world cups", "who won the most",
                     "most titles"],
        "ar": "البرازيل أكثر منتخب فاز بكأس العالم بـ 5 ألقاب. لكن معلوماتي تركّز على نسخة 2026 تحديدًا.",
        "en": "Brazil has won the most World Cups — 5 titles. My data focuses on the 2026 edition specifically though.",
    },
    {
        "name": "last_winner",
        "keywords": ["who won the last world cup", "last world cup winner",
                     "who won the 2022", "آخر منتخب فاز", "اخر منتخب فاز",
                     "من فاز بكأس العالم الماضي", "من فاز 2022"],
        "ar": "الأرجنتين فازت بكأس العالم 2022 في قطر. معلوماتي الرئيسية عن نسخة 2026.",
        "en": "Argentina won the 2022 World Cup in Qatar. My main data is about the 2026 edition.",
    },
    {
        "name": "ticket_price",
        "keywords": ["سعر التذكره", "سعر التذكرة", "سعر تذكره", "كم التذاكر",
                     "ticket price", "how much ticket", "cost of ticket"],
        "ar": "ما عندي معلومات عن أسعار التذاكر — تتغيّر حسب المباراة والفئة. تابع الموقع الرسمي fifa.com للتفاصيل.",
        "en": "I don't have ticket price details — they vary by match and category. Check fifa.com for the latest.",
    },
    {
        "name": "tv_broadcast",
        "keywords": ["وين تنعرض", "وين تتعرض", "اي قناه تنقل", "أي قناة تنقل",
                     "tv channel", "where to watch", "broadcast", "which channel"],
        "ar": "حقوق البث تختلف من دولة لدولة، تابع القنوات الرسمية في منطقتك للتأكد.",
        "en": "Broadcast rights vary by country — check the official broadcasters in your region.",
    },
]

# Pre-normalise the keyword lists so we don't pay that cost per request.
for _intent in SOCIAL_INTENTS:
    _intent["_normalised"] = [_normalize(k) for k in _intent["keywords"]]


def _match_social_intent(message: str, language: str) -> str | None:
    """Return a canned response if the message matches a social intent.

    Matches via either:
      * Substring containment (fast path for short messages).
      * Token-set containment (so "انت مين" and "مين انت" both hit the same
        keyword "مين انت" regardless of word order).
    """
    norm = _normalize(message)
    if not norm:
        return None
    msg_tokens = set(norm.split())
    for intent in SOCIAL_INTENTS:
        for kw in intent["_normalised"]:
            if not kw:
                continue
            if kw in norm:
                return intent[language] if language in ("ar", "en") else intent["en"]
            kw_tokens = set(kw.split())
            if kw_tokens and kw_tokens <= msg_tokens:
                return intent[language] if language in ("ar", "en") else intent["en"]
    return None


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat payload."""

    message: str = Field(..., min_length=1, description="The user's question.")
    language: Literal["ar", "en"] = Field(
        default="en", description="Reply language hint."
    )
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    session_id: str | None = Field(
        default=None,
        description=(
            "Opaque client-generated UUID. When supplied, the server threads "
            "the last few user/assistant turns through the prompt so follow-up "
            "questions resolve correctly."
        ),
    )


class SourceChunk(BaseModel):
    """One retrieval result surfaced back to the caller."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Outgoing chat payload."""

    answer: str
    sources: list[SourceChunk]
    language: Literal["ar", "en"]
    model: str


# ---------------------------------------------------------------------------
# Model singleton (loaded once at startup)
# ---------------------------------------------------------------------------

class ModelHandle:
    """Wraps the tokenizer + (possibly adapter-equipped) Mistral model."""

    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.model: Any = None
        self.label: str = "uninitialized"
        self._lock: asyncio.Lock = asyncio.Lock()

    def load(self) -> None:
        """Load the Mistral model. Retrieval lives in a sidecar — see RETRIEVAL."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        if (ADAPTER_DIR / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM

            logger.info("Loading PEFT model from %s …", ADAPTER_DIR)
            tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), use_fast=True)
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(ADAPTER_DIR),
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.float16,
                token=settings.hf_token,
            )
            self.label = f"{BASE_MODEL_NAME}+lora"
        else:
            logger.warning(
                "No adapter at %s — serving raw base model. Run "
                "`python -m src.training.finetune` to enable fine-tuned answers.",
                ADAPTER_DIR,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME, token=settings.hf_token, use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb,
                device_map="auto",
                token=settings.hf_token,
                torch_dtype=torch.float16,
            )
            self.label = BASE_MODEL_NAME

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model.config.use_cache = True
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        logger.info("Model ready (%s).", self.label)

    async def generate(self, prompt_messages: list[dict[str, str]]) -> str:
        """Render the chat messages and generate an assistant reply.

        Serialised by an asyncio lock so concurrent /chat requests don't
        thrash the GPU.
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        async with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=TEMPERATURE > 0,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # greedy beam=1 is the cheapest path
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def stream(self, prompt_messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield assistant tokens as they're produced (perceived-latency win).

        Internally runs ``model.generate`` on a worker thread with a
        ``TextIteratorStreamer`` and pulls tokens into the asyncio loop via
        a thread-pool executor. Serialised by the same asyncio lock as the
        non-streaming path so concurrent requests don't race on the GPU.
        """
        import torch
        from transformers import TextIteratorStreamer

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        async with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=60.0,
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                streamer=streamer,
            )

            def _run_generate() -> None:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)

            thread = Thread(target=_run_generate, daemon=True)
            thread.start()

            loop = asyncio.get_running_loop()
            try:
                while True:
                    chunk = await loop.run_in_executor(None, _next_chunk, streamer)
                    if chunk is _STREAM_END:
                        break
                    if chunk:
                        yield chunk
            finally:
                thread.join(timeout=5.0)


_STREAM_END: object = object()
def _next_chunk(streamer: Any) -> Any:
    """Blocking ``next()`` adapter for use inside ``run_in_executor``."""
    try:
        return next(streamer)
    except StopIteration:
        return _STREAM_END


HANDLE: ModelHandle = ModelHandle()


# ---------------------------------------------------------------------------
# Conversation memory — bounded in-memory LRU keyed by client session_id.
# Persisting across restarts is intentionally out of scope; the UI also keeps
# the visible transcript in localStorage, so a server restart just loses the
# model-side context window, not the user's history.
# ---------------------------------------------------------------------------

class SessionStore:
    """Per-session conversation history with LRU eviction."""

    def __init__(self, max_sessions: int = MAX_SESSIONS,
                 max_turns: int = MAX_TURNS_PER_SESSION) -> None:
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._max_messages = max_turns * 2  # user + assistant per turn

    async def history(self, session_id: str | None) -> list[dict[str, str]]:
        """Return a copy of this session's history (oldest → newest)."""
        if not session_id:
            return []
        async with self._lock:
            msgs = self._sessions.get(session_id)
            if msgs is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(msgs)

    async def append(self, session_id: str | None,
                     user_msg: str, assistant_msg: str) -> None:
        """Record one user/assistant turn."""
        if not session_id:
            return
        async with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    self._sessions.popitem(last=False)  # evict oldest
                self._sessions[session_id] = []
            self._sessions[session_id].extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ])
            # Bound history length per session.
            if len(self._sessions[session_id]) > self._max_messages:
                self._sessions[session_id] = self._sessions[session_id][-self._max_messages:]
            self._sessions.move_to_end(session_id)

    async def clear(self, session_id: str | None) -> None:
        """Drop a session entirely (used by the New Chat button)."""
        if not session_id:
            return
        async with self._lock:
            self._sessions.pop(session_id, None)


SESSIONS: SessionStore = SessionStore()


# ---------------------------------------------------------------------------
# Retrieval sidecar (subprocess) — keeps the MiniLM encoder out of this
# bnb-tainted process. See src/rag/retrieval_sidecar.py.
# ---------------------------------------------------------------------------

class RetrievalSidecar:
    """Owns the long-lived sidecar subprocess and serialises requests to it."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    def start(self) -> None:
        """Spawn the sidecar and wait for its ``{"ready": true}`` handshake.

        Uses raw byte pipes and forces ``PYTHONIOENCODING=utf-8`` in the
        child env because Windows' default subprocess pipe codepage replaces
        non-Latin-1 characters with ``?`` even when ``text=True,
        encoding='utf-8'`` is set on the Python side.
        """
        import os

        logger.info("Spawning retrieval sidecar …")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.rag.retrieval_sidecar"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit — sidecar's logs flow to uvicorn's stderr
            bufsize=0,    # unbuffered binary pipe
            env=env,
        )
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Retrieval sidecar failed to start (no handshake).")
        msg = json.loads(line.decode("utf-8"))
        if not msg.get("ready"):
            raise RuntimeError(f"Retrieval sidecar handshake failed: {msg}")
        logger.info("Retrieval sidecar ready.")

    def stop(self) -> None:
        """Best-effort shutdown of the sidecar."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.write((json.dumps({"action": "shutdown"}) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            logger.exception("Sidecar shutdown raised — terminating.")
            self._proc.terminate()

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Send one retrieval request to the sidecar and parse the response."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Retrieval sidecar is not running.")

        assert self._proc.stdin is not None and self._proc.stdout is not None
        payload = (
            json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False) + "\n"
        ).encode("utf-8")

        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: (self._proc.stdin.write(payload), self._proc.stdin.flush()),  # type: ignore[union-attr]
            )
            line_bytes = await loop.run_in_executor(None, self._proc.stdout.readline)

        if not line_bytes:
            raise RuntimeError("Retrieval sidecar closed unexpectedly.")
        msg = json.loads(line_bytes.decode("utf-8"))
        if "error" in msg:
            raise RuntimeError(f"Retrieval sidecar error: {msg['error']}")

        return [
            RetrievalHit(
                text=h["text"],
                source=h["source"],
                score=float(h["score"]),
                metadata=dict(h.get("metadata", {})),
            )
            for h in msg.get("hits", [])
        ]


RETRIEVAL: RetrievalSidecar = RetrievalSidecar()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_context(hits: list[RetrievalHit]) -> str:
    """Concatenate retrieved chunks (truncated) into a prompt-ready block."""
    lines: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        snippet = h.text.strip().replace("\n", " ")
        chunk = f"[{i}] ({h.source}) {snippet}"
        if used + len(chunk) > MAX_CONTEXT_CHARS:
            break
        lines.append(chunk)
        used += len(chunk)
    return "\n".join(lines) if lines else "(no relevant context retrieved)"


def _build_messages(
    question: str,
    language: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the message list passed to the chat template.

    Past turns from ``history`` are spliced between the system prompt and
    the current user turn. Only the current user turn carries the retrieved
    context block — past assistant messages stay context-free to keep the
    prompt short.
    """
    system = SYSTEM_PROMPT_AR if language == "ar" else SYSTEM_PROMPT_EN
    context_label = "السياق" if language == "ar" else "Context"
    question_label = "السؤال" if language == "ar" else "Question"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"{context_label}:\n{context}\n\n{question_label}: {question}",
    })
    return messages


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Spawn the retrieval sidecar, then load the Mistral model."""
    RETRIEVAL.start()
    HANDLE.load()
    try:
        yield
    finally:
        RETRIEVAL.stop()


app = FastAPI(
    title="World Cup 2026 Chatbot",
    description="Bilingual RAG + QLoRA Mistral assistant.",
    version="0.1.0",
    lifespan=_lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe + which model is currently serving."""
    return {
        "status": "ok",
        "model": HANDLE.label,
        "adapter_loaded": HANDLE.label.endswith("+lora"),
    }


@app.get("/", response_model=None)
def root() -> Any:
    """Serve the chat UI when present, else a small JSON banner."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "World Cup 2026 Chatbot API — POST /chat to ask a question."}


_FOLLOWUP_AR_PREFIXES: tuple[str, ...] = ("و", "وش", "ومين", "ومتى", "وكم",
                                           "ولماذا", "وكيف", "وين", "بس", "طيب")
_FOLLOWUP_EN_PREFIXES: tuple[str, ...] = ("and ", "what about", "how about",
                                           "so ", "ok ", "okay ", "what if")
_FOLLOWUP_PRONOUNS: tuple[str, ...] = (
    " هي ", " هو ", " هم ", " هن ", " فيها ", " فيه ", " معه ", " معها ",
    " it ", " they ", " them ", " its ", " their ",
)


def _looks_like_followup(message: str, has_history: bool) -> bool:
    """Heuristic: is this message a referent-laden follow-up to prior turn?"""
    if not has_history:
        return False
    m = message.strip()
    if len(m) < 30:
        return True
    low = m.lower()
    if any(low.startswith(p) for p in _FOLLOWUP_EN_PREFIXES):
        return True
    if any(m.startswith(p) for p in _FOLLOWUP_AR_PREFIXES):
        return True
    spaced = " " + low + " "
    if any(pn in spaced for pn in _FOLLOWUP_PRONOUNS):
        return True
    return False


async def _resolve_chat(req: ChatRequest) -> tuple[str, list[RetrievalHit], bool]:
    """Common pre-LLM path used by both /chat and /chat/stream.

    Returns ``(answer_or_empty, strong_hits, needs_llm)``:
      * If a non-LLM answer is available (refusal, social intent, static
        fact, or weak retrieval), it's in ``answer_or_empty`` and
        ``needs_llm`` is False.
      * Otherwise ``answer_or_empty`` is empty, ``strong_hits`` are the
        retrieved chunks, and ``needs_llm`` is True.

    History-aware retrieval: when the current message looks like a
    follow-up ("ومين أقوى فيها؟"), we expand the retrieval query with the
    last user turn so the retriever finds chunks about the *right entity*.
    Generation still sees only the original message (plus history) so the
    answer stays focused.
    """
    fallback = FALLBACK_AR if req.language == "ar" else FALLBACK_EN

    if len(req.message.strip()) < MIN_QUESTION_CHARS:
        return fallback, [], False

    social = _match_social_intent(req.message, req.language)
    if social is not None:
        return social, [], False

    static_answer = static_facts.lookup(req.message, req.language)
    if static_answer is not None:
        return static_answer, [], False

    # History-aware retrieval query.
    history = await SESSIONS.history(req.session_id)
    retrieval_query = req.message
    if _looks_like_followup(req.message, bool(history)):
        # Find the most recent USER turn in history.
        prior_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            None,
        )
        if prior_user:
            retrieval_query = f"{prior_user} {req.message}"
            logger.info("history-aware query: %r", retrieval_query[:120])

    hits = await RETRIEVAL.retrieve(retrieval_query, top_k=req.top_k)
    strong_hits = [h for h in hits if h.score >= RETRIEVAL_SCORE_THRESHOLD]

    if not strong_hits:
        return fallback, hits[:3], False

    return "", strong_hits, True


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """RAG-augmented chat with optional conversation memory.

    Path through the layers (see ``_resolve_chat``):
      1. Reject too-short input.
      2. Social intent match.
      3. Static-fact match.
      4. RAG retrieve → if no strong hit, refuse.
      5. Else: build prompt with prior turns + retrieved context, generate.
    """
    if HANDLE.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        canned, hits, needs_llm = await _resolve_chat(req)
    except Exception as exc:
        logger.exception("Resolve failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not needs_llm:
        if req.session_id and canned:
            await SESSIONS.append(req.session_id, req.message, canned)
        return ChatResponse(
            answer=canned, sources=_to_source_chunks(hits),
            language=req.language, model=HANDLE.label,
        )

    history = await SESSIONS.history(req.session_id)
    context_block = _format_context(hits)
    messages = _build_messages(req.message, req.language, context_block, history)

    try:
        answer = await HANDLE.generate(messages)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    if _is_hallucination(answer):
        logger.warning("Suppressed hallucinated answer: %r", answer[:120])
        answer = FALLBACK_AR if req.language == "ar" else FALLBACK_EN

    if req.session_id:
        await SESSIONS.append(req.session_id, req.message, answer)

    return ChatResponse(
        answer=answer, sources=_to_source_chunks(hits),
        language=req.language, model=HANDLE.label,
    )


def _to_source_chunks(hits: list[RetrievalHit]) -> list[SourceChunk]:
    return [SourceChunk(text=h.text, source=h.source, score=h.score, metadata=h.metadata)
            for h in hits]


def _sse_event(payload: dict[str, Any]) -> bytes:
    """Encode one Server-Sent Events frame."""
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-sent events stream — same logic as /chat but tokens flow live.

    Event shapes (all wrapped in ``data: {...}``):
        ``{"type": "meta", "sources": [...], "model": "..."}``
        ``{"type": "token", "text": "..."}``  (zero or more)
        ``{"type": "done"}``                  (always last)
        ``{"type": "error", "message": "..."}``
    """
    if HANDLE.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            canned, hits, needs_llm = await _resolve_chat(req)
        except Exception as exc:
            logger.exception("Resolve failed (stream)")
            yield _sse_event({"type": "error", "message": str(exc)})
            yield _sse_event({"type": "done"})
            return

        yield _sse_event({
            "type": "meta",
            "sources": [
                {"text": h.text, "source": h.source, "score": h.score,
                 "metadata": h.metadata}
                for h in hits
            ],
            "model": HANDLE.label,
            "language": req.language,
        })

        if not needs_llm:
            yield _sse_event({"type": "token", "text": canned})
            if req.session_id and canned:
                await SESSIONS.append(req.session_id, req.message, canned)
            yield _sse_event({"type": "done"})
            return

        history = await SESSIONS.history(req.session_id)
        context_block = _format_context(hits)
        messages = _build_messages(req.message, req.language, context_block, history)

        collected: list[str] = []
        try:
            async for chunk in HANDLE.stream(messages):
                collected.append(chunk)
                yield _sse_event({"type": "token", "text": chunk})
        except Exception as exc:
            logger.exception("Streaming generation failed")
            yield _sse_event({"type": "error", "message": str(exc)})
            yield _sse_event({"type": "done"})
            return

        full_answer = "".join(collected).strip()
        if _is_hallucination(full_answer):
            fallback = FALLBACK_AR if req.language == "ar" else FALLBACK_EN
            logger.warning("Suppressed hallucinated streamed answer.")
            yield _sse_event({"type": "replace", "text": fallback})
            full_answer = fallback

        if req.session_id and full_answer:
            await SESSIONS.append(req.session_id, req.message, full_answer)

        yield _sse_event({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",          # disable proxy buffering
        },
    )


class SessionClearRequest(BaseModel):
    session_id: str


@app.post("/session/clear")
async def session_clear(req: SessionClearRequest) -> dict[str, str]:
    """Drop a session's server-side history (UI's New Chat button hits this)."""
    await SESSIONS.clear(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "src.api.chat:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
