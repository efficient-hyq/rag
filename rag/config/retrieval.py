from __future__ import annotations

import os
from dataclasses import dataclass

from rag.config.common import (
    EmbeddingConfig,
    LLMServiceConfig,
    PathConfig,
    get_bool_env,
    load_embedding_config_from_env,
    load_llm_service_config_from_env,
    load_path_config_from_env,
)


@dataclass(frozen=True)
class RetrievalRouteConfig:
    """多路召回配置。"""

    content_top_k: int = 8
    summary_top_k: int = 6
    bm25_top_k: int = 8
    rewrite_content_top_k: int = 4
    rewrite_summary_top_k: int = 3
    rewrite_bm25_top_k: int = 4
    neighbor_expand_enabled: bool = True
    neighbor_radius: int = 1
    center_top_k: int = 5


@dataclass(frozen=True)
class RetrievalRewriteConfig:
    """Query Rewrite 配置。"""

    enabled: bool = True
    model: str = "qwen3.6-plus"
    limit: int = 3


@dataclass(frozen=True)
class RetrievalRankingConfig:
    """在线重排配置。"""

    llm_enabled: bool = True
    llm_model: str = "qwen3.6-plus"
    llm_top_n: int = 10
    final_top_n: int = 5


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """最终答案生成配置。"""

    enabled: bool = True
    model: str = "qwen3.6-plus"
    context_top_k: int = 4
    max_context_chars: int = 1200


@dataclass(frozen=True)
class QueryConfig:
    """在线查询配置，按查询流程聚合相关参数。"""

    paths: PathConfig
    llm: LLMServiceConfig
    embedding: EmbeddingConfig
    routes: RetrievalRouteConfig
    rewrite: RetrievalRewriteConfig
    ranking: RetrievalRankingConfig
    answer: AnswerGenerationConfig

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载在线查询配置。"""
        paths = load_path_config_from_env()
        llm = load_llm_service_config_from_env()
        return cls(
            paths=paths,
            llm=llm,
            embedding=load_embedding_config_from_env(llm),
            routes=RetrievalRouteConfig(
                content_top_k=int(os.getenv("RAG_RETRIEVAL_CONTENT_TOP_K", "8")),
                summary_top_k=int(os.getenv("RAG_RETRIEVAL_SUMMARY_TOP_K", "6")),
                bm25_top_k=int(os.getenv("RAG_RETRIEVAL_BM25_TOP_K", "8")),
                rewrite_content_top_k=int(os.getenv("RAG_RETRIEVAL_REWRITE_CONTENT_TOP_K", "4")),
                rewrite_summary_top_k=int(os.getenv("RAG_RETRIEVAL_REWRITE_SUMMARY_TOP_K", "3")),
                rewrite_bm25_top_k=int(os.getenv("RAG_RETRIEVAL_REWRITE_BM25_TOP_K", "4")),
                neighbor_expand_enabled=get_bool_env("RAG_RETRIEVAL_NEIGHBOR_ENABLED", True),
                neighbor_radius=int(os.getenv("RAG_RETRIEVAL_NEIGHBOR_RADIUS", "1")),
                center_top_k=int(os.getenv("RAG_RETRIEVAL_CENTER_TOP_K", "5")),
            ),
            rewrite=RetrievalRewriteConfig(
                enabled=get_bool_env("RAG_RETRIEVAL_REWRITE_ENABLED", True),
                model=os.getenv("RAG_QUERY_REWRITE_MODEL", "qwen3.6-plus"),
                limit=int(os.getenv("RAG_RETRIEVAL_REWRITE_LIMIT", "3")),
            ),
            ranking=RetrievalRankingConfig(
                llm_enabled=get_bool_env("RAG_RERANK_LLM_ENABLED", True),
                llm_model=os.getenv("RAG_RERANK_LLM_MODEL", "qwen3.6-plus"),
                llm_top_n=int(os.getenv("RAG_RERANK_LLM_TOP_N", "10")),
                final_top_n=int(os.getenv("RAG_RERANK_FINAL_TOP_N", "5")),
            ),
            answer=AnswerGenerationConfig(
                enabled=get_bool_env("RAG_ANSWER_ENABLED", True),
                model=os.getenv("RAG_ANSWER_MODEL", "qwen3.6-plus"),
                context_top_k=int(os.getenv("RAG_ANSWER_CONTEXT_TOP_K", "4")),
                max_context_chars=int(os.getenv("RAG_ANSWER_MAX_CONTEXT_CHARS", "1200")),
            ),
        )
