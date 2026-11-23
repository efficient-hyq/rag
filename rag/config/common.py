from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def project_root() -> Path:
    """仓库根目录。本文件在 rag/config 下，向上三级才是项目根。"""
    return Path(__file__).resolve().parents[2]


def default_storage_dir() -> Path:
    """索引、预处理产物和 chunk 都放在项目根目录的 storage 下。"""
    return project_root() / "storage"


def resolve_project_path(value: str | Path | None, *, default: str = "storage") -> Path:
    """把相对路径解析到项目根目录，绝对路径保持不变。

    这样无论从哪个工作目录启动命令，./storage 都落到项目根的 storage，
    而不会落到 rag/cli/storage。
    """
    if value is None or str(value).strip() == "":
        raw = default
    else:
        raw = str(value).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def default_docs_dir() -> Path:
    """优先使用带图片说明的 Markdown，没有时回退到清洗结果。"""
    storage = default_storage_dir()
    enriched = storage / "enriched_markdown"
    if enriched.is_dir():
        return enriched
    return storage / "cleaned_markdown"


@dataclass(frozen=True)
class PathConfig:
    """项目输入输出路径配置。"""

    docs_dir: Path = field(default_factory=default_docs_dir)
    storage_dir: Path = field(default_factory=default_storage_dir)


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
    batch_size: int = 20


def load_path_config_from_env() -> PathConfig:
    """从环境变量加载项目路径配置。"""
    storage_dir = resolve_project_path(os.getenv("RAG_STORAGE_DIR"))
    docs_env = os.getenv("RAG_DOCS_DIR")
    if docs_env and docs_env.strip():
        docs_dir = resolve_project_path(docs_env)
    else:
        enriched = storage_dir / "enriched_markdown"
        docs_dir = enriched if enriched.is_dir() else storage_dir / "cleaned_markdown"
    return PathConfig(docs_dir=docs_dir, storage_dir=storage_dir)


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
        batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "20")),
    )


def get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
