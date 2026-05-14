# World Cup 2026 RAG Chatbot — task runner.
# Requires `uv` (https://docs.astral.sh/uv/) and GNU make.
# On Windows, run targets via Git Bash, WSL, or `make` from chocolatey.

.PHONY: help install install-dev lint format typecheck test \
        serve-api serve-streamlit \
        download-data ingest train-allam train-qwen evaluate \
        clean clean-cache

help:
	@echo "Available targets:"
	@echo "  install          Sync runtime deps with uv"
	@echo "  install-dev      Sync runtime + dev deps"
	@echo "  lint             Run ruff"
	@echo "  format           Run black + ruff --fix"
	@echo "  typecheck        Run mypy"
	@echo "  test             Run pytest"
	@echo "  serve-api        Run FastAPI on :8000"
	@echo "  serve-streamlit  Run the Streamlit MVP UI"
	@echo "  download-data    Fetch Kaggle + Wikipedia data"
	@echo "  ingest           Build SQLite DB and Chroma index from raw data"
	@echo "  train-allam      QLoRA fine-tune ALLaM-7B (Arabic)"
	@echo "  train-qwen       QLoRA fine-tune Qwen2.5-7B (English)"
	@echo "  evaluate         Run the RAGAS evaluation suite"
	@echo "  clean            Remove caches and build artifacts"

install:
	uv sync --no-dev

install-dev:
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run black src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

serve-api:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

serve-streamlit:
	uv run streamlit run src/ui/streamlit_app.py

download-data:
	uv run python -m scripts.download_data

ingest:
	uv run python -m src.data.ingestion

train-allam:
	uv run python -m src.training.train_allam

train-qwen:
	uv run python -m src.training.train_qwen

evaluate:
	uv run python -m src.evaluation.ragas_eval

clean: clean-cache
	rm -rf build dist *.egg-info

clean-cache:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
