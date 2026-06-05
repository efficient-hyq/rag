from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.config import BuildIndexConfig
from rag.indexing.document_loader import (
    MarkdownDocumentDiff,
    collect_current_markdown_files,
    collect_current_markdown_state,
    compute_document_content_hash,
    diff_markdown_documents,
    load_documents_from_files,
    normalize_doc_key,
)
from rag.indexing.embedding_client import (
    build_openai_compatible_embedder,
    embed_nodes_with_checkpoint,
    embed_texts,
)
from rag.indexing.markdown_chunker import split_documents
from rag.indexing.preview_renderer import write_document_chunk_previews
from rag.indexing.semantic_annotator import SemanticAnnotator
from rag.indexing.storage_indexer import IndexResult, MultiRouteIndexer
from rag.shared.checkpoints import CheckpointStore
from rag.shared.logging_utils import log_phase


def build_markdown_diff(docs_root: Path, checkpoint: CheckpointStore) -> MarkdownDocumentDiff:
    previous_state = checkpoint.load_document_index_state()
    current_hashes = collect_current_markdown_state(docs_root)
    return diff_markdown_documents(previous_state, current_hashes)


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
    """执行默认文档级增量建库流程。"""
    logger = logging.getLogger("rag.pipeline")
    cfg = config or BuildIndexConfig.from_env()
    docs_root = Path(docs_dir) if docs_dir is not None else cfg.paths.docs_dir
    storage_root = Path(storage_dir) if storage_dir is not None else cfg.paths.storage_dir
    checkpoint = CheckpointStore(storage_root)
    writer = indexer or MultiRouteIndexer(storage_root)

    with log_phase(logger, "识别增量文档", docs_dir=str(docs_root)):
        previous_state = checkpoint.load_document_index_state()
        current_files = collect_current_markdown_files(docs_root)
        current_hashes = {
            doc_key: compute_document_content_hash(path)
            for doc_key, path in current_files.items()
        }
        diff = diff_markdown_documents(previous_state, current_hashes)
    logger.info(
        "增量识别完成 | 新增=%s | 变更=%s | 删除=%s | 复用=%s",
        len(diff.added),
        len(diff.changed),
        len(diff.deleted),
        len(diff.unchanged),
    )

    stale_doc_keys = diff.changed | diff.deleted
    stale_node_ids = gather_stale_node_ids(previous_state, stale_doc_keys)
    if stale_node_ids:
        with log_phase(logger, "清理旧索引", node_count=len(stale_node_ids)):
            writer.delete_nodes(stale_node_ids)
            checkpoint.remove_node_records(stale_node_ids)
    for doc_key in stale_doc_keys:
        writer.remove_metadata_shard(doc_key)

    rebuild_keys = sorted(diff.added | diff.changed)
    if not rebuild_keys:
        writer.rebuild_metadata_snapshot()
        writer.rebuild_bm25_from_metadata_snapshot()
        checkpoint.save_document_index_state(
            build_next_document_state(previous_state, current_hashes, diff.unchanged, [], docs_root)
        )
        logger.info("离线索引完成 | 本次无新增或变更文档，已刷新兼容索引快照")
        return IndexResult(
            node_count=0,
            content_collection="content_vec",
            summary_collection="summary_vec",
            bm25_path=writer.bm25_path,
            metadata_path=writer.metadata_path,
        )

    rebuild_files = [current_files[key] for key in rebuild_keys]
    with log_phase(logger, "加载增量文档", docs=len(rebuild_files)):
        documents = load_documents_from_files(rebuild_files, docs_root)
    if not documents:
        raise RuntimeError("存在新增或变更 Markdown 文档，但未加载到可重建文档")
    logger.info("增量文档加载完成 | 文档数=%s", len(documents))

    with log_phase(
        logger,
        "切分 chunk",
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    ):
        nodes = split_documents(documents, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    logger.info("切分完成 | chunk数=%s", len(nodes))

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
    checkpoint.write_chunks(nodes)
    write_document_chunk_previews(nodes, storage_root)

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
    write_document_chunk_previews(nodes, storage_root)
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

    with log_phase(logger, "多路入库", nodes=len(nodes), storage_dir=str(storage_root)):
        result = writer.index(nodes, content_embeddings, summary_embeddings, root_doc_dir=docs_root)
    checkpoint.save_document_index_state(
        build_next_document_state(previous_state, current_hashes, diff.unchanged, nodes, docs_root)
    )
    logger.info(
        "离线索引完成 | chunk数=%s | metadata=%s | bm25=%s",
        result.node_count,
        result.metadata_path,
        result.bm25_path,
    )
    return result


def gather_stale_node_ids(previous_state: dict[str, Any], doc_keys: set[str]) -> set[str]:
    previous_docs = previous_state.get("docs", {})
    if not isinstance(previous_docs, dict):
        return set()
    node_ids: set[str] = set()
    for doc_key in doc_keys:
        record = previous_docs.get(doc_key)
        if not isinstance(record, dict):
            continue
        node_ids.update(str(node_id) for node_id in record.get("node_ids", []) if node_id)
    return node_ids


def build_next_document_state(
    previous_state: dict[str, Any],
    current_hashes: dict[str, str],
    unchanged_keys: set[str],
    rebuilt_nodes: list[Any],
    docs_root: Path,
) -> dict[str, Any]:
    previous_docs = previous_state.get("docs", {})
    if not isinstance(previous_docs, dict):
        previous_docs = {}
    rebuilt_node_ids = group_node_ids_by_doc_key(rebuilt_nodes, docs_root)
    updated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    docs: dict[str, dict[str, Any]] = {}
    for doc_key in sorted(current_hashes):
        if doc_key in unchanged_keys and isinstance(previous_docs.get(doc_key), dict):
            record = dict(previous_docs[doc_key])
            record["content_hash"] = current_hashes[doc_key]
            docs[doc_key] = record
            continue

        docs[doc_key] = {
            "content_hash": current_hashes[doc_key],
            "node_ids": rebuilt_node_ids.get(doc_key, []),
            "updated_at": updated_at,
        }
    return {"docs": docs}


def group_node_ids_by_doc_key(nodes: list[Any], docs_root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_node_metadata(metadata, docs_root)
        node_id = str(getattr(node, "node_id", None) or getattr(node, "id_", None) or "")
        if doc_key and node_id:
            groups.setdefault(doc_key, []).append(node_id)
    return groups


def doc_key_from_node_metadata(metadata: dict[str, Any], docs_root: Path) -> str:
    for key in ("cleaned_markdown_relative_path", "doc_id", "file_path", "filename"):
        value = metadata.get(key)
        if not value:
            continue
        raw_path = str(value).replace("\\", "/").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            return normalize_doc_key(path, docs_root)
        except ValueError:
            return path.as_posix().lower()
    return "unknown"
