"""Smoke tests for src.config."""

from pathlib import Path

from src.config import PROJECT_ROOT, settings


def test_project_root_is_absolute() -> None:
    assert PROJECT_ROOT.is_absolute()


def test_paths_under_project_root() -> None:
    for p in (
        settings.paths.data,
        settings.paths.csvs,
        settings.paths.wikipedia,
        settings.paths.sqlite_db,
        settings.paths.chroma,
        settings.paths.models,
    ):
        assert isinstance(p, Path)
        assert PROJECT_ROOT in p.parents or p == PROJECT_ROOT


def test_model_identifiers_nonempty() -> None:
    assert settings.models.allam_base
    assert settings.models.qwen_base
    assert settings.models.embedding
    assert settings.models.reranker


def test_rag_defaults() -> None:
    assert settings.rag.chunk_size > 0
    assert settings.rag.chunk_overlap < settings.rag.chunk_size
    assert settings.rag.top_k_rerank <= settings.rag.top_k_retrieval
