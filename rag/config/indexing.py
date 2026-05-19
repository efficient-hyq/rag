from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag.config.common import (
    EmbeddingConfig,
    LLMServiceConfig,
    PathConfig,
    load_embedding_config_from_env,
    load_llm_service_config_from_env,
    load_path_config_from_env,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """文档切分配置。"""

    chunk_size: int = 512
    chunk_overlap: int = 100


@dataclass(frozen=True)
class AnnotationConfig:
    """离线语义标注配置。"""

    model: str = "qwen3.6-plus"
    workers: int = 5
    prompt_path: Path = Path("prompts/annotation_v2.md")
    prompt_version: str = "annotation_v2"


@dataclass(frozen=True)
class BuildIndexConfig:
    """离线建库配置，按建库流程聚合相关参数。"""

    paths: PathConfig
    llm: LLMServiceConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    annotation: AnnotationConfig

    @classmethod
    def from_env(cls) -> "BuildIndexConfig":
        """从环境变量加载离线建库配置。"""
        paths = load_path_config_from_env()
        llm = load_llm_service_config_from_env()
        return cls(
            paths=paths,
            llm=llm,
            embedding=load_embedding_config_from_env(llm),
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            ),
            annotation=AnnotationConfig(
                model=os.getenv("RAG_ANNOTATOR_MODEL", "qwen3.6-plus"),
                workers=int(os.getenv("RAG_ANNOTATION_WORKERS", "5")),
                prompt_path=Path(os.getenv("RAG_ANNOTATION_PROMPT_PATH", "prompts/annotation_v2.md")),
                prompt_version=os.getenv("RAG_ANNOTATION_PROMPT_VERSION", "annotation_v2"),
            ),
        )
