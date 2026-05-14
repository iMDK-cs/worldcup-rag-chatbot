"""Project-wide configuration.

All filesystem paths, model identifiers, and retrieval parameters are defined here.
Override any value through environment variables (see ``.env.example``).

Usage:
    from src.config import settings

    db_path = settings.paths.sqlite_db
    embed_model_id = settings.models.embedding
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Paths(BaseSettings):
    """Canonical filesystem layout. Override any path via ``WC_PATH_<NAME>``."""

    model_config = SettingsConfigDict(env_prefix="WC_PATH_", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    csvs: Path = PROJECT_ROOT / "data" / "raw" / "csvs"
    wikipedia: Path = PROJECT_ROOT / "data" / "raw" / "wikipedia"
    kaggle: Path = PROJECT_ROOT / "data" / "raw" / "kaggle"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    translations: Path = PROJECT_ROOT / "data" / "translations"
    synthetic: Path = PROJECT_ROOT / "data" / "synthetic"
    models: Path = PROJECT_ROOT / "models"
    chroma: Path = PROJECT_ROOT / "chroma_db"
    sqlite_db: Path = PROJECT_ROOT / "data" / "processed" / "worldcup.db"

    def ensure(self) -> None:
        """Create every directory in the layout if it does not already exist."""
        for p in (
            self.data,
            self.raw,
            self.csvs,
            self.wikipedia,
            self.kaggle,
            self.processed,
            self.translations,
            self.synthetic,
            self.models,
            self.chroma,
        ):
            p.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseSettings):
    """LLM, embedding, reranker, and QLoRA hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="WC_MODEL_", extra="ignore")

    # Base models
    allam_base: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    qwen_base: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"

    # QLoRA / quantization
    quantization_bits: Literal[4, 8] = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Inference
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class RAGConfig(BaseSettings):
    """Chunking and retrieval parameters."""

    model_config = SettingsConfigDict(env_prefix="WC_RAG_", extra="ignore")

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 4
    sql_synthesize_response: bool = True


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    wikipedia_user_agent: str = Field(
        default="WorldCupRAG/0.1.0 (contact@example.com)",
        alias="WIKIPEDIA_USER_AGENT",
    )

    # Runtime
    app_env: Literal["dev", "prod"] = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Sub-configurations
    paths: Paths = Field(default_factory=Paths)
    models: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance."""
    return Settings()


settings: Settings = get_settings()
