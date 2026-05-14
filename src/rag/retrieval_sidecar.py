"""Long-lived stdio sidecar for retrieval — keeps the encoder out of the
bitsandbytes-tainted API process.

Why: on Windows, loading 4-bit Mistral with bitsandbytes in the same process
as the MiniLM encoder silently degrades the encoder's outputs (cosine
similarities drop by ~30%). This sidecar runs in its own Python interpreter
so the encoder is never touched by bnb.

Protocol (newline-delimited JSON, stdin/stdout):

    request  → {"query": str, "top_k": int}
    response → {"hits": [{"text": str, "source": str, "score": float,
                          "metadata": dict}, ...]}

Errors are surfaced as ``{"error": "..."}``. The process exits when stdin
closes or it receives ``{"action": "shutdown"}``.

Spawn with:
    python -m src.rag.retrieval_sidecar
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.rag.numpy_retriever import retrieve

logger = logging.getLogger(__name__)


def _serve() -> None:
    """Read newline-delimited JSON from stdin, write JSON responses to stdout."""
    # Encourage line buffering so the parent sees each response immediately.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    # Warm up — load the encoder + index now so the first /chat request is fast.
    try:
        retrieve("warmup", top_k=1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sidecar warmup failed")
        print(json.dumps({"error": f"warmup failed: {exc}"}), flush=True)
        return

    # Ready signal — parent waits for this before forwarding queries.
    print(json.dumps({"ready": True}), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"bad json: {exc}"}), flush=True)
            continue

        if req.get("action") == "shutdown":
            break

        query = req.get("query")
        top_k = int(req.get("top_k", 5))
        if not isinstance(query, str) or not query:
            print(json.dumps({"error": "missing 'query' field"}), flush=True)
            continue

        try:
            hits = retrieve(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrieval failed")
            print(json.dumps({"error": f"retrieval failed: {exc}"}), flush=True)
            continue

        response: dict[str, Any] = {
            "hits": [
                {
                    "text": h.text,
                    "source": h.source,
                    "score": h.score,
                    "metadata": h.metadata,
                }
                for h in hits
            ]
        }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # Log to stderr so it doesn't pollute the stdout protocol channel.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    _serve()
