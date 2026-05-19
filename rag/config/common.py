from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class PathConfig:
    """项目输入输出路径配置。"""

    docs_dir: Path = Path("./storage/cleaned_markdown")
    storage_dir: Path = Path("./storage")


@dataclass(frozen=True)
class LLMServiceConfig:
    """LLM 服务配置。"""

    api_key: str | None = None
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 服务配置。"""

    api_key: str | None = None
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    model: str = "text-embedding-v4"
    batch_size: int = 10


def load_path_config_from_env() -> PathConfig:
    """从环境变量加载项目路径配置。"""
    return PathConfig(
        docs_dir=Path(os.getenv("RAG_DOCS_DIR", "./storage/cleaned_markdown")),
        storage_dir=Path(os.getenv("RAG_STORAGE_DIR", "./storage")),
    )


def load_llm_service_config_from_env() -> LLMServiceConfig:
    """从环境变量加载 LLM 服务配置。"""
    return LLMServiceConfig(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL),
    )


def load_embedding_config_from_env(llm: LLMServiceConfig) -> EmbeddingConfig:
    """从环境变量加载 Embedding 配置，并在未单独设置时回退到 LLM 配置。"""
    return EmbeddingConfig(
        api_key=os.getenv("RAG_EMBEDDING_API_KEY"),
        base_url=os.getenv("RAG_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
        model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4"),
        batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10")),
    )


def get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
