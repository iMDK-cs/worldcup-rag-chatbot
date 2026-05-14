# Mundial Chatbot — World Cup 2026 RAG Assistant

A bilingual (Arabic / English) retrieval-augmented chatbot focused on the
FIFA World Cup 2026 — schedule, host cities, groups, knockout structure,
match probabilities, and tournament history. Built around a 4-bit
QLoRA-fine-tuned `mistralai/Mistral-7B-Instruct-v0.3` model, a
numpy-backed multilingual retriever, and a FastAPI streaming server.

| | |
|---|---|
| **Base model** | `mistralai/Mistral-7B-Instruct-v0.3` (NF4 4-bit) |
| **Adapter** | LoRA r=16, α=32, dropout=0.05 (7 projections) |
| **Training set** | 6,949 instruction/output pairs (AR + EN + Saudi colloquial) |
| **Embedder** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Retriever** | numpy cosine over 7,158 indexed chunks |
| **API** | FastAPI + SSE streaming + asyncio session memory |
| **UI** | Single-file HTML, AR/EN toggle, dark/light themes |

---

## Benchmark

20 prompts across four categories. Latencies in seconds; hit-rate is the
share of answers whose final text contains at least one expected keyword.

| Category | N | Hit rate | TTFT p50 | Total p50 | Total p95 |
|---|---|---|---|---|---|
| Social (greetings, identity) | 4 | 100% | 2.03 | 2.03 | 2.05 |
| Static facts (titles, format) | 6 | 100% | 2.02 | 2.02 | 2.04 |
| RAG (in-corpus 2026 facts) | 7 | 86% | 3.05 | 3.53 | 7.76 |
| Out-of-scope (refuse politely) | 3 | 100% | 2.02 | 2.02 | 2.04 |
| **Overall** | **20** | **95%** | **2.04** | **2.04** | **5.42** |

Source: `src/evaluation/benchmark.py` → `models/benchmark_report.json`.

---

## Architecture

```
                    +-----------------+
   user query  -->  | /chat or stream |
                    +--------+--------+
                             |
                  +----------v-----------+
                  | 1. social intents    |  short-circuit greetings, identity,
                  |    (token-set fuzzy) |  thanks, player participation
                  +----------+-----------+
                             | miss
                  +----------v-----------+
                  | 2. static facts      |  authoritative answers for
                  |    (titles, players, |  history / format the RAG corpus
                  |     format, rounds)  |  does not cover
                  +----------+-----------+
                             | miss
                  +----------v-----------+      +-----------------------+
                  | 3. retrieval sidecar |----->| stdio subprocess      |
                  |    (history-aware    |      |   MiniLM encoder      |
                  |     query rewrite)   |      |   numpy cosine        |
                  +----------+-----------+      +-----------------------+
                             | top score >= 0.55
                  +----------v-----------+
                  | 4. Mistral-7B + LoRA |  generates grounded answer
                  |    streaming via SSE |  with conversation history
                  +----------+-----------+
                             | hallucination filter
                             v
                       SSE token stream
```

### Multi-layer answer routing

Each request walks four layers in order; the first match wins:

1. **Social intents** — token-set fuzzy matching over greetings, identity,
   thanks, and well-known player questions (Messi, Ronaldo, Mbappé, etc.).
   Word-order independent: "انت مين" and "مين انت" both match.
2. **Static facts** — regex + alias dictionary for historical World Cup
   titles per nation, the tournament format breakdown, the knockout
   structure, and detailed player profiles. Anything the RAG corpus
   doesn't cover.
3. **RAG + LLM** — multilingual MiniLM retrieves the top 5 chunks across
   schedule CSV, host-cities CSV, probability CSV, and the 6,949-pair Q&A
   dataset. The LLM generates a grounded answer using the retrieved
   context plus the prior conversation history.
4. **Polite refusal** — when nothing crosses the retrieval threshold,
   the server returns a fixed fallback in the user's language.

### Conversation memory

A bounded in-memory `SessionStore` (LRU, 500 sessions, 6 turns each)
keeps the last few user/assistant turns per `session_id`. The client
generates a UUID once and persists it in `localStorage`. History is
threaded into the LLM prompt; **history-aware retrieval** also concatenates
the previous user turn into the retrieval query when the current message
looks like a follow-up ("ومين أقوى فيها؟", "and what about...").

### Defence against hallucination

- 4-bit base + LoRA at `temperature=0` (pure greedy).
- Retrieval threshold of 0.55; below it the request never reaches the LLM.
- Post-generation regex filter against known fabricated patterns
  ("قاعدة الـ N ثوانٍ", "Captain Only rule", etc.) — replaces with the
  polite fallback.
- 36 fabricated rows filtered out of the training set at clean time.

### Retrieval sidecar (Windows)

On Windows, `bitsandbytes` and `chromadb` / `sentence-transformers` cannot
share a single Python process — the latter silently corrupt the encoder's
outputs once 4-bit Mistral is loaded. The retrieval encoder therefore runs
in its own subprocess (`src/rag/retrieval_sidecar.py`), communicating with
the API over a UTF-8 byte stdio pipe.

---

## Project layout

```
worldcup-rag-chatbot/
├── data/
│   ├── raw/csvs/                        FIFA2026_schedule, host_cities, probabilities
│   ├── translations/                    teams / venues / cities Arabic <-> English
│   └── synthetic/qa_train_clean.jsonl   6,949 deduplicated Q&A pairs
├── models/                              LoRA adapter + training_loss.png (gitignored)
├── src/
│   ├── api/chat.py                      FastAPI app, SSE streaming, sessions
│   ├── config.py                        pydantic Settings
│   ├── data/
│   │   └── static_facts.py              titles, players, format
│   ├── evaluation/
│   │   ├── benchmark.py                 latency + accuracy battery
│   │   └── compare.py                   RAG-only vs RAG + LoRA eval
│   ├── rag/
│   │   ├── numpy_retriever.py           in-process cosine retrieval
│   │   ├── retrieval_sidecar.py         stdio subprocess host for the encoder
│   │   └── pipeline.py                  one-shot document collection
│   └── training/
│       ├── finetune.py                  QLoRA fine-tune (HF + PEFT + TRL)
│       ├── merge_datasets.py            dedup multi-source merge
│       ├── clean_dataset.py             hallucination filter
│       └── qa_generator_v2.py           batched Anthropic-API generator
├── static/index.html                    self-contained chat UI
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with at least 10 GB VRAM (RTX 3080 or better) for inference
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A Hugging Face token with access to `mistralai/Mistral-7B-Instruct-v0.3`

### Install

```bash
uv sync
cp .env.example .env
# Fill HF_TOKEN (required) and ANTHROPIC_API_KEY (only if regenerating data).
```

### Build the retrieval index

```bash
uv run python -m src.rag.numpy_retriever --build
```

This embeds every chunk in `data/synthetic/qa_train_clean.jsonl` plus the
three source CSVs and writes the index to `data/chroma_db/numpy_index/`.
Runs on CPU (the encoder is ~120 MB).

### Train the adapter (optional — the model can run without one)

```bash
uv run python -m src.training.finetune
```

Trains for two epochs at LR 1e-4 with a 5 % held-out validation split and
saves the LoRA adapter to `models/mistral-7b-worldcup/`. Roughly 70 minutes
on an RTX 3080.

### Run the server

```bash
uv run uvicorn src.api.chat:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/> — the UI loads with dark/light and AR/EN toggles.

---

## API

### `POST /chat`

JSON in, JSON out. Honours conversation history when `session_id` is supplied.

Request:

```json
{
  "message": "متى تبدأ بطولة كأس العالم 2026؟",
  "language": "ar",
  "top_k": 5,
  "session_id": "client-generated-uuid"
}
```

Response:

```json
{
  "answer": "11 يونيو 2026.",
  "sources": [
    {"text": "Q: متى تبدأ البطولة؟\nA: 11 يونيو 2026.",
     "source": "qa_ar", "score": 0.892, "metadata": {...}}
  ],
  "language": "ar",
  "model": "mistralai/Mistral-7B-Instruct-v0.3+lora"
}
```

### `POST /chat/stream`

Same payload, returns a Server-Sent Events stream:

```
data: {"type":"meta","sources":[...],"model":"...","language":"..."}
data: {"type":"token","text":"11 "}
data: {"type":"token","text":"يونيو 2026."}
data: {"type":"done"}
```

### `POST /session/clear`

```json
{ "session_id": "client-generated-uuid" }
```

Drops the session's server-side history. The UI's *new chat* button hits
this and then rotates the local session identifier.

### `GET /health`

Liveness + which model is currently serving (`+lora` suffix when the
adapter is loaded).

---

## Data sources

Committed:

- `host_cities.csv` — 16 host cities (11 USA, 3 Mexico, 2 Canada)
- `FIFA2026_schedule.csv` — every match with date and stadium
- `future_match_probabilities_baseline.csv` — Elo-derived group-stage
  win/draw/loss probabilities
- `data/translations/*.json` — bilingual mappings for teams, cities,
  venues
- `data/synthetic/qa_train_clean.jsonl` — 6,949 instruction/output pairs

Regenerable (gitignored):

- `data/raw/wikipedia/` and `data/raw/kaggle/` — fetched on demand
- `data/chroma_db/numpy_index/` — produced by `python -m src.rag.numpy_retriever --build`
- `models/mistral-7b-worldcup/` — produced by `python -m src.training.finetune`

---

## Regenerating the Q&A dataset

The bundled `qa_train_clean.jsonl` is the canonical training set. To
rebuild it from scratch:

```bash
# 1. Generate new pairs (requires ANTHROPIC_API_KEY)
uv run python -m src.training.qa_generator_v2 --target 5000

# 2. Merge any other JSONL sources and de-duplicate
uv run python -m src.training.merge_datasets

# 3. Strip rows with banned prefixes or fabricated rules
uv run python -m src.training.clean_dataset

# 4. Rebuild the retrieval index
uv run python -m src.rag.numpy_retriever --build
```

---

## Benchmark and evaluation

```bash
# Live API latency + hit-rate
uv run python -m src.evaluation.benchmark

# RAG-only vs RAG + fine-tuned LoRA (10 bilingual prompts)
uv run python -m src.evaluation.compare
```

Outputs `models/benchmark_report.json` and `models/eval_results.json`.

---

## Known limitations

- General football knowledge outside 2026 is limited to whatever is hard-coded in `static_facts.py`. Anything else triggers a polite refusal.
- The fine-tune amplifies tournament-specific tone but is not a general assistant; complex multi-step reasoning is out of scope.
- Probability answers cite the bundled baseline Elo model, not live betting markets.
- Conversation memory is in-process; restarting the server drops it.
- The retrieval-sidecar workaround targets Windows; on Linux the encoder
  can run in the same process without segfaulting.

---

## License

MIT
