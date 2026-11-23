"""离线建库：只消费 prepare_chunks 已经切好的 chunk。

这一步先给每个 chunk 写定位上下文，再做标注、摘要、上下文嵌入、依赖图和向量入库。
不再读取 Markdown 重新切分，替换和删除也按 chunk 指纹判断。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.config import BuildIndexConfig
from rag.config.common import resolve_project_path
from rag.indexing.chunk_context import (
    CONTEXT_VERSION,
    build_retrieval_text,
    chunk_content_hash,
    document_content_hash,
    load_prepared_chunk_records,
    records_to_nodes,
)
from rag.indexing.dependency_graph import DependencyAggregationResult, DependencyCardStore, DocumentDependencyAggregator
from rag.indexing.embedding_client import (
    build_openai_compatible_embedder,
    embed_nodes_with_checkpoint,
    embed_texts,
)
from rag.indexing.preview_renderer import write_document_chunk_previews
from rag.indexing.publication import prepare_index_workspace, publish_index
from rag.indexing.rebuild_decision import (
    CHUNK_STATE_SCHEMA,
    ChunkRebuildDecision,
    compute_chunk_rebuild_decision,
    previous_node_ids,
    prompt_fingerprint,
)
from rag.indexing.semantic_annotator import SemanticAnnotator
from rag.indexing.situating_context import SituatingContextGenerator, load_situating_prompt, load_whole_documents
from rag.indexing.storage_indexer import IndexResult, MultiRouteIndexer, doc_key_from_metadata
from rag.shared.checkpoints import CheckpointStore, node_key, text_hash
from rag.shared.logging_utils import log_phase


def build_offline_index(
    chunks_path: str | Path | None = None,
    storage_dir: str | Path | None = None,
    annotator: Any | None = None,
    embedder: Any | None = None,
    content_embedder: Any | None = None,
    summary_embedder: Any | None = None,
    indexer: MultiRouteIndexer | None = None,
    dependency_aggregator: Any | None = None,
    situating_generator: Any | None = None,
    config: BuildIndexConfig | None = None,
    _managed_workspace: bool = False,
) -> IndexResult:
    """对已切分 chunk 做增量标注、依赖分析、上下文嵌入和入库。"""
    logger = logging.getLogger("rag.pipeline")
    cfg = config or BuildIndexConfig.from_env()
    storage_root = (
        resolve_project_path(storage_dir)
        if storage_dir is not None
        else Path(cfg.paths.storage_dir)
    )
    chunks_file = (
        resolve_project_path(chunks_path)
        if chunks_path is not None
        else storage_root / "prepared_chunks" / "chunks.jsonl"
    )

    if indexer is None:
        workspace = prepare_index_workspace(storage_root)
        logger.info(
            "使用索引构建工作区 | workspace=%s | chunks=%s",
            workspace,
            chunks_file,
        )
        try:
            result = build_offline_index(
                chunks_path=chunks_file,
                storage_dir=workspace,
                annotator=annotator,
                embedder=embedder,
                content_embedder=content_embedder,
                summary_embedder=summary_embedder,
                indexer=MultiRouteIndexer(workspace),
                dependency_aggregator=dependency_aggregator,
                situating_generator=situating_generator,
                config=cfg,
                _managed_workspace=True,
            )
            final_dir = publish_index(storage_root, workspace)
        except Exception:
            logger.exception("离线索引未发布，构建工作区已保留 | workspace=%s", workspace)
            raise
        logger.info("索引已发布 | workspace=%s | final=%s", workspace, final_dir)
        return replace(
            result,
            bm25_path=final_dir / result.bm25_path.name,
            metadata_path=final_dir / result.metadata_path.name,
        )

    checkpoint = CheckpointStore(storage_root)
    writer = indexer
    records = load_prepared_chunk_records(chunks_file)
    nodes = records_to_nodes(records)
    logger.info(
        "已读取切分 chunk | path=%s | chunk数=%s | storage=%s | managed=%s",
        chunks_file,
        len(records),
        storage_root,
        _managed_workspace,
    )

    annotation_prompt = _read_prompt(cfg.annotation.prompt_path)
    dependency_prompt = _read_prompt(cfg.dependency.prompt_path)
    situating_prompt = load_situating_prompt(cfg.situating.prompt_path)
    annotation_hash = prompt_fingerprint(annotation_prompt)
    dependency_hash = prompt_fingerprint(dependency_prompt)
    situating_hash = prompt_fingerprint(situating_prompt)
    documents = load_whole_documents(nodes, cfg.paths.docs_dir)
    document_hashes = {doc_key: text_hash(document) for doc_key, document in documents.items()}
    grouped = _group_chunks(records, annotation_hash)
    previous_state = checkpoint.load_document_index_state()
    decision = compute_chunk_rebuild_decision(
        grouped["chunk_hashes"],
        grouped["doc_hashes"],
        grouped["chunk_docs"],
        previous_state,
        annotation_prompt_version=cfg.annotation.prompt_version,
        annotation_prompt_hash=annotation_hash,
        dependency_prompt_version=cfg.dependency.prompt_version,
        dependency_prompt_hash=dependency_hash,
        embedding_model=cfg.embedding.model,
        storage_root=storage_root,
        situating_prompt_version=cfg.situating.prompt_version,
        situating_prompt_hash=situating_hash,
        situating_doc_hashes=document_hashes,
    )
    logger.info(
        "chunk 重建决策 | 新增=%d | 替换=%d | 删除=%d | 不变=%d | 重标注=%d | 重嵌入=%d | 重依赖文档=%d",
        len(decision.added),
        len(decision.replaced),
        len(decision.deleted),
        len(decision.unchanged),
        len(decision.need_reannotate),
        len(decision.need_reembed),
        len(decision.need_redependency_docs),
    )
    logger.info("决策原因: %s", decision.reason)

    manifest = _build_manifest(
        cfg,
        chunks_file,
        storage_root,
        annotation_prompt,
        dependency_prompt,
        situating_prompt,
    )
    if decision.needs_no_rebuild:
        checkpoint.write_manifest(manifest)
        logger.info("离线索引完成 | 本次 chunk 无变化")
        return _empty_result(writer)

    checkpoint.write_manifest(manifest)
    checkpoint.write_chunks(nodes)
    _apply_situating_contexts(
        nodes,
        documents,
        document_hashes,
        cfg,
        situating_prompt,
        situating_hash,
        checkpoint,
        situating_generator,
    )
    _annotate_changed_chunks(
        nodes,
        decision,
        cfg,
        annotation_prompt,
        checkpoint,
        annotator,
    )
    write_document_chunk_previews(nodes, storage_root)
    dependency_result = _aggregate_changed_documents(
        nodes,
        decision,
        grouped["doc_hashes"],
        storage_root,
        cfg,
        dependency_prompt,
        checkpoint,
        dependency_aggregator,
    )

    must_reindex = bool(decision.need_reembed or decision.added or decision.replaced or decision.deleted)
    if must_reindex:
        content_embeddings, summary_embeddings = _embed_chunks(
            nodes,
            cfg,
            checkpoint,
            embedder,
            content_embedder,
            summary_embedder,
        )
        stale_ids = decision.deleted or (previous_node_ids(previous_state) - {node_key(node) for node in nodes})
        if stale_ids:
            with log_phase(logger, "清理旧 chunk", node_count=len(stale_ids)):
                writer.delete_nodes(stale_ids)
                if cfg.dependency.vector_enabled:
                    writer.delete_dependency_nodes(stale_ids)
        with log_phase(logger, "多路入库", nodes=len(nodes), storage_dir=str(storage_root)):
            result = writer.index(nodes, content_embeddings, summary_embeddings, root_doc_dir=storage_root)
        if cfg.dependency.vector_enabled:
            _index_dependency_vectors(
                nodes,
                dependency_result.documents if dependency_result is not None else {},
                storage_root,
                writer,
                content_embedder or embedder,
                cfg,
            )
    else:
        result = _empty_result(writer)
        result = replace(result, node_count=len(nodes))

    next_state = _build_chunk_state(
        records,
        grouped,
        cfg,
        annotation_hash,
        dependency_hash,
        situating_hash,
        document_hashes,
        previous_state,
        decision,
    )
    checkpoint.save_document_index_state(next_state)
    checkpoint.remove_node_records(decision.deleted)
    checkpoint.remove_dependency_documents(decision.deleted_docs)
    checkpoint.prune_dependency_cards(grouped["doc_hashes"])
    logger.info(
        "离线索引完成 | chunk数=%s | metadata=%s | bm25=%s",
        result.node_count,
        result.metadata_path,
        result.bm25_path,
    )
    return result


def _annotate_changed_chunks(
    nodes: list[Any],
    decision: ChunkRebuildDecision,
    cfg: BuildIndexConfig,
    annotation_prompt: str,
    checkpoint: CheckpointStore,
    annotator: Any | None,
) -> None:
    logger = logging.getLogger("rag.pipeline")
    pending = [node for node in nodes if node_key(node) in decision.need_reannotate]
    if not pending:
        if annotator is None:
            semantic = _build_annotator(cfg, annotation_prompt)
            missing = _apply_cached_annotations(semantic, nodes, checkpoint)
            if missing:
                raise RuntimeError("存在未标注的 chunk，无法继续入库: " + ", ".join(missing[:5]))
        return

    semantic = annotator or _build_annotator(cfg, annotation_prompt)
    with log_phase(logger, "语义标注", nodes=len(pending), checkpoint_enabled=annotator is None):
        if annotator is None:
            annotation_result = semantic.annotate_nodes(pending, checkpoint=checkpoint)
        else:
            annotation_result = semantic.annotate_nodes(pending)
    if annotation_result.failed_count > 0:
        logger.error("标注完成但存在失败 | 失败数=%s", annotation_result.failed_count)
        raise RuntimeError(
            f"标注阶段存在 {annotation_result.failed_count} 个失败 chunk，"
            "已完成整批标注，但不会继续向量化和入库。"
        )
    if annotator is None:
        missing = _apply_cached_annotations(semantic, nodes, checkpoint)
        if missing:
            raise RuntimeError("标注检查点缺少 chunk，无法继续入库: " + ", ".join(missing[:5]))


def _aggregate_changed_documents(
    nodes: list[Any],
    decision: ChunkRebuildDecision,
    doc_hashes: dict[str, str],
    storage_root: Path,
    cfg: BuildIndexConfig,
    dependency_prompt: str,
    checkpoint: CheckpointStore,
    dependency_aggregator: Any | None,
) -> DependencyAggregationResult | None:
    logger = logging.getLogger("rag.pipeline")
    store = DependencyCardStore(storage_root)
    if not decision.need_redependency_docs and not decision.deleted_docs:
        return None
    selected = [
        node
        for node in nodes
        if doc_key_from_metadata(dict(getattr(node, "metadata", {}) or {}), storage_root)
        in decision.need_redependency_docs
    ]
    if not selected:
        store.update_documents({}, decision.deleted_docs)
        return DependencyAggregationResult(documents={}, failed_docs=[])

    aggregator = dependency_aggregator or DocumentDependencyAggregator(
        api_key=cfg.llm.api_key or "",
        base_url=cfg.llm.base_url,
        model=cfg.dependency.model,
        prompt=dependency_prompt,
        prompt_version=cfg.dependency.prompt_version,
    )
    with log_phase(logger, "文档依赖聚合", docs=len(decision.need_redependency_docs), nodes=len(selected)):
        if dependency_aggregator is None:
            dependency_result = aggregator.aggregate_documents(
                selected,
                doc_hashes,
                storage_root,
                checkpoint=checkpoint,
            )
        else:
            dependency_result = aggregator.aggregate_documents(selected, doc_hashes, storage_root)
    store.update_documents(dependency_result.documents, decision.deleted_docs)
    if dependency_result.failed_docs:
        raise RuntimeError(
            "依赖图聚合存在未成功文档，不发布当前索引批次: " + ", ".join(dependency_result.failed_docs)
        )
    return dependency_result


def _embed_chunks(
    nodes: list[Any],
    cfg: BuildIndexConfig,
    checkpoint: CheckpointStore,
    embedder: Any | None,
    content_embedder: Any | None,
    summary_embedder: Any | None,
) -> tuple[list[list[float]], list[list[float]]]:
    logger = logging.getLogger("rag.pipeline")
    content_texts = [
        build_retrieval_text(str(getattr(node, "text", "") or ""), dict(getattr(node, "metadata", {}) or {}))
        for node in nodes
    ]
    summary_texts = [str((getattr(node, "metadata", {}) or {}).get("summary") or "") for node in nodes]
    if embedder is not None or content_embedder is not None or summary_embedder is not None:
        content_model, summary_model = _resolve_explicit_embedders(embedder, content_embedder, summary_embedder)
        with log_phase(logger, "上下文嵌入", nodes=len(nodes), checkpoint_enabled=False):
            content_embeddings = embed_texts(content_model, content_texts)
        with log_phase(logger, "摘要向量化", nodes=len(nodes), checkpoint_enabled=False):
            summary_embeddings = embed_texts(summary_model, summary_texts)
        return content_embeddings, summary_embeddings

    if not cfg.embedding.api_key or not cfg.embedding.base_url:
        raise ValueError(
            "缺少向量化 OpenAI 兼容接口配置，请设置 RAG_EMBEDDING_API_KEY "
            "和 RAG_EMBEDDING_BASE_URL，或提供 DASHSCOPE_API_KEY"
        )
    model = build_openai_compatible_embedder(cfg.embedding.api_key, cfg.embedding.base_url, cfg.embedding.model)
    with log_phase(logger, "上下文嵌入", nodes=len(nodes), checkpoint_enabled=True):
        content_embeddings = embed_nodes_with_checkpoint(
            model,
            nodes,
            content_texts,
            checkpoint,
            "content",
            batch_size=cfg.embedding.batch_size,
        )
    with log_phase(logger, "摘要向量化", nodes=len(nodes), checkpoint_enabled=True):
        summary_embeddings = embed_nodes_with_checkpoint(
            model,
            nodes,
            summary_texts,
            checkpoint,
            "summary",
            batch_size=cfg.embedding.batch_size,
        )
    return content_embeddings, summary_embeddings


def _index_dependency_vectors(
    nodes: list[Any],
    documents: dict[str, dict[str, Any]],
    storage_root: Path,
    writer: MultiRouteIndexer,
    embedder: Any | None,
    cfg: BuildIndexConfig,
) -> None:
    from rag.indexing.dependency_graph import build_dependency_vector_records

    logger = logging.getLogger("rag.pipeline")
    records = build_dependency_vector_records(nodes, documents, storage_root)
    if not records:
        return
    model = embedder
    if model is None:
        model = build_openai_compatible_embedder(cfg.embedding.api_key or "", cfg.embedding.base_url, cfg.embedding.model)
    with log_phase(logger, "dependency 向量化", records=len(records)):
        embeddings = embed_texts(model, [str(record.get("text") or "") for record in records])
    writer.index_dependency_vectors(records, embeddings)


def _group_chunks(records: list[dict[str, Any]], annotation_prompt_hash: str) -> dict[str, Any]:
    chunk_hashes: dict[str, str] = {}
    chunk_docs: dict[str, str] = {}
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for record in records:
        node_id = str(record["node_id"])
        content_hash = chunk_content_hash(record)
        doc_key = doc_key_from_metadata(record, Path("."))
        chunk_hashes[node_id] = content_hash
        chunk_docs[node_id] = doc_key
        grouped.setdefault(doc_key, []).append((int(record.get("chunk_index") or 0), node_id, content_hash))

    doc_hashes: dict[str, str] = {}
    doc_node_ids: dict[str, list[str]] = {}
    for doc_key, items in grouped.items():
        items.sort()
        doc_node_ids[doc_key] = [node_id for _, node_id, _ in items]
        doc_hashes[doc_key] = document_content_hash(
            [content_hash for _, _, content_hash in items],
            annotation_prompt_hash,
        )
    return {
        "chunk_hashes": chunk_hashes,
        "chunk_docs": chunk_docs,
        "doc_hashes": doc_hashes,
        "doc_node_ids": doc_node_ids,
    }


def _apply_situating_contexts(
    nodes: list[Any],
    documents: dict[str, str],
    document_hashes: dict[str, str],
    cfg: BuildIndexConfig,
    situating_prompt: str,
    situating_hash: str,
    checkpoint: CheckpointStore,
    situating_generator: Any | None,
) -> None:
    """先写定位上下文，再让标注读取同一份检索文本。"""
    logger = logging.getLogger("rag.pipeline")
    generator = situating_generator or _build_situating_generator(cfg, situating_prompt, situating_hash)
    with log_phase(logger, "片段定位上下文", nodes=len(nodes)):
        generator.apply(nodes, documents, document_hashes, checkpoint)


def _build_situating_generator(
    cfg: BuildIndexConfig,
    prompt: str,
    prompt_hash: str,
) -> SituatingContextGenerator:
    if not cfg.llm.api_key:
        raise ValueError("缺少 LLM API Key，请设置 DASHSCOPE_API_KEY")
    return SituatingContextGenerator(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.situating.model or "",
        prompt=prompt,
        prompt_hash=prompt_hash,
        max_workers=cfg.situating.workers,
    )


def _build_chunk_state(
    records: list[dict[str, Any]],
    grouped: dict[str, Any],
    cfg: BuildIndexConfig,
    annotation_hash: str,
    dependency_hash: str,
    situating_hash: str,
    document_hashes: dict[str, str],
    previous_state: dict[str, Any],
    decision: ChunkRebuildDecision,
) -> dict[str, Any]:
    previous_docs = previous_state.get("docs") if isinstance(previous_state.get("docs"), dict) else {}
    updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    changed_docs = set(decision.need_redependency_docs) | {
        grouped["chunk_docs"].get(node_id, "")
        for node_id in decision.added | decision.replaced | decision.need_reannotate | decision.need_reembed
    }
    docs: dict[str, dict[str, Any]] = {}
    for doc_key, node_ids in sorted(grouped["doc_node_ids"].items()):
        previous = previous_docs.get(doc_key) if isinstance(previous_docs.get(doc_key), dict) else {}
        keep_time = doc_key not in changed_docs and previous.get("updated_at")
        docs[doc_key] = {
            "content_hash": grouped["doc_hashes"][doc_key],
            "node_ids": list(node_ids),
            "node_count": len(node_ids),
            "updated_at": str(previous.get("updated_at")) if keep_time else updated_at,
        }
    chunks = {
        str(record["node_id"]): {
            "content_hash": grouped["chunk_hashes"][str(record["node_id"])],
            "doc_key": grouped["chunk_docs"][str(record["node_id"])],
        }
        for record in records
    }
    return {
        "schema_version": CHUNK_STATE_SCHEMA,
        "context_version": CONTEXT_VERSION,
        "annotation_prompt_version": cfg.annotation.prompt_version,
        "annotation_prompt_hash": annotation_hash,
        "dependency_prompt_version": cfg.dependency.prompt_version,
        "dependency_prompt_hash": dependency_hash,
        "situating_prompt_version": cfg.situating.prompt_version,
        "situating_prompt_hash": situating_hash,
        "situating_doc_hashes": document_hashes,
        "embedding_model": cfg.embedding.model,
        "total_chunks": len(chunks),
        "total_docs": len(docs),
        "total_nodes": len(chunks),
        "chunks": chunks,
        "docs": docs,
    }


def _build_manifest(
    cfg: BuildIndexConfig,
    chunks_file: Path,
    storage_root: Path,
    annotation_prompt: str,
    dependency_prompt: str,
    situating_prompt: str,
) -> dict[str, Any]:
    return {
        "chunks_path": str(chunks_file),
        "storage_dir": str(storage_root),
        "context_version": CONTEXT_VERSION,
        "annotator_model": cfg.annotation.model,
        "annotation_prompt_version": cfg.annotation.prompt_version,
        "annotation_prompt_hash": prompt_fingerprint(annotation_prompt),
        "dependency_prompt_version": cfg.dependency.prompt_version,
        "dependency_prompt_hash": prompt_fingerprint(dependency_prompt),
        "situating_prompt_version": cfg.situating.prompt_version,
        "situating_prompt_hash": prompt_fingerprint(situating_prompt),
        "situating_model": cfg.situating.model,
        "embedding_model": cfg.embedding.model,
        "dependency_vector_enabled": cfg.dependency.vector_enabled,
    }


def _build_annotator(cfg: BuildIndexConfig, prompt: str) -> SemanticAnnotator:
    if not cfg.llm.api_key:
        raise ValueError("缺少 LLM API Key，请设置 DASHSCOPE_API_KEY")
    return SemanticAnnotator(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.annotation.model,
        max_workers=cfg.annotation.workers,
        prompt=prompt,
        prompt_version=cfg.annotation.prompt_version,
    )


def _apply_cached_annotations(
    annotator: SemanticAnnotator,
    nodes: list[Any],
    checkpoint: CheckpointStore,
) -> list[str]:
    cached = checkpoint.load_annotations()
    missing: list[str] = []
    for node in nodes:
        key = annotator._cache_key(node)
        annotation = cached.get(key)
        if not annotation:
            missing.append(node_key(node))
            continue
        annotator._apply_annotation(node, annotation)
    return missing


def _resolve_explicit_embedders(
    embedder: Any | None,
    content_embedder: Any | None,
    summary_embedder: Any | None,
) -> tuple[Any, Any]:
    if embedder is not None:
        return embedder, embedder
    if content_embedder is None or summary_embedder is None:
        raise ValueError("content_embedder 和 summary_embedder 必须同时提供")
    return content_embedder, summary_embedder


def _empty_result(writer: MultiRouteIndexer) -> IndexResult:
    return IndexResult(
        node_count=0,
        content_collection="content_vec",
        summary_collection="summary_vec",
        bm25_path=writer.bm25_path,
        metadata_path=writer.metadata_path,
    )


def _read_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
