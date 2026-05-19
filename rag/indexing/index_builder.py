from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.config import BuildIndexConfig
from rag.indexing.document_loader import load_documents
from rag.indexing.embedding_client import (
    build_openai_compatible_embedder,
    embed_nodes_with_checkpoint,
    embed_texts,
)
from rag.indexing.markdown_chunker import split_documents
from rag.indexing.preview_renderer import write_chunk_preview
from rag.indexing.semantic_annotator import SemanticAnnotator
from rag.indexing.storage_indexer import IndexResult, MultiRouteIndexer
from rag.shared.checkpoints import CheckpointStore
from rag.shared.logging_utils import log_phase


def build_offline_index(
        docs_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        annotator: Any | None = None,
        embedder: Any | None = None,
        content_embedder: Any | None = None,
        summary_embedder: Any | None = None,
        indexer: MultiRouteIndexer | None = None,
        config: BuildIndexConfig | None = None,
) -> IndexResult:
    """执行离线建库主流程：加载、切分、标注、向量化、多路入库。"""
    logger = logging.getLogger("rag.pipeline")
    cfg = config or BuildIndexConfig.from_env()
    docs_root = Path(docs_dir) if docs_dir is not None else cfg.paths.docs_dir
    storage_root = Path(storage_dir) if storage_dir is not None else cfg.paths.storage_dir
    checkpoint = CheckpointStore(storage_root)

    with log_phase(logger, "加载文档", docs_dir=str(docs_root)):
        documents = load_documents(docs_root)
    logger.info("加载完成 | 文档数=%s", len(documents))

    with log_phase(
        logger,
        "切分 chunk",
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    ):
        nodes = split_documents(documents, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    logger.info("切分完成 | chunk数=%s", len(nodes))

    checkpoint.write_chunks(nodes)
    checkpoint.write_manifest(
        {
            "docs_dir": str(docs_root),
            "storage_dir": str(storage_root),
            "chunk_size": cfg.chunking.chunk_size,
            "chunk_overlap": cfg.chunking.chunk_overlap,
            "annotator_model": cfg.annotation.model,
            "annotation_prompt_version": cfg.annotation.prompt_version,
            "annotation_checkpoint_enabled": annotator is None,
            "embedding_model": cfg.embedding.model,
        }
    )
    write_chunk_preview(nodes, storage_root)

    semantic_annotator = annotator or SemanticAnnotator(
        api_key=cfg.llm.api_key or "",
        base_url=cfg.llm.base_url,
        model=cfg.annotation.model,
        max_workers=cfg.annotation.workers,
        prompt=cfg.annotation.prompt_path.read_text(encoding="utf-8")
        if cfg.annotation.prompt_path.exists()
        else None,
        prompt_version=cfg.annotation.prompt_version,
    )
    with log_phase(logger, "语义标注", nodes=len(nodes), checkpoint_enabled=annotator is None):
        if annotator is None:
            annotation_result = semantic_annotator.annotate_nodes(nodes, checkpoint=checkpoint)
        else:
            annotation_result = semantic_annotator.annotate_nodes(nodes)
    nodes = annotation_result.nodes
    write_chunk_preview(nodes, storage_root)
    if annotation_result.failed_count > 0:
        logger.error("标注完成但存在失败 | 失败数=%s", annotation_result.failed_count)
        raise RuntimeError(
            f"标注阶段存在 {annotation_result.failed_count} 个失败 chunk，"
            "已完成整批标注，但不会继续向量化和入库。"
        )

    if embedder is not None:
        content_model = embedder
        summary_model = embedder
    elif content_embedder is not None or summary_embedder is not None:
        content_model = content_embedder
        summary_model = summary_embedder
        if content_model is None or summary_model is None:
            raise ValueError("content_embedder 和 summary_embedder 必须同时提供")
    else:
        if not cfg.embedding.api_key or not cfg.embedding.base_url:
            raise ValueError(
                "缺少向量化 OpenAI 兼容接口配置，请设置 RAG_EMBEDDING_API_KEY "
                "和 RAG_EMBEDDING_BASE_URL，或提供 DASHSCOPE_API_KEY"
            )
        content_model = build_openai_compatible_embedder(
            cfg.embedding.api_key,
            cfg.embedding.base_url,
            cfg.embedding.model,
        )
        summary_model = content_model

    if embedder is not None or content_embedder is not None or summary_embedder is not None:
        with log_phase(logger, "content 向量化", nodes=len(nodes), checkpoint_enabled=False):
            content_embeddings = embed_texts(content_model, [node.text for node in nodes])
        with log_phase(logger, "summary 向量化", nodes=len(nodes), checkpoint_enabled=False):
            summary_embeddings = embed_texts(
                summary_model,
                [str(node.metadata.get("summary") or "") for node in nodes],
            )
    else:
        with log_phase(logger, "content 向量化", nodes=len(nodes), checkpoint_enabled=True):
            content_embeddings = embed_nodes_with_checkpoint(
                content_model,
                nodes,
                [node.text for node in nodes],
                checkpoint,
                "content",
                batch_size=cfg.embedding.batch_size,
            )
        with log_phase(logger, "summary 向量化", nodes=len(nodes), checkpoint_enabled=True):
            summary_embeddings = embed_nodes_with_checkpoint(
                summary_model,
                nodes,
                [str(node.metadata.get("summary") or "") for node in nodes],
                checkpoint,
                "summary",
                batch_size=cfg.embedding.batch_size,
            )

    writer = indexer or MultiRouteIndexer(storage_root)
    with log_phase(logger, "多路入库", nodes=len(nodes), storage_dir=str(storage_root)):
        result = writer.index(nodes, content_embeddings, summary_embeddings)
    logger.info(
        "离线索引完成 | chunk数=%s | metadata=%s | bm25=%s",
        result.node_count,
        result.metadata_path,
        result.bm25_path,
    )
    return result
