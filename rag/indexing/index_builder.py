from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.config import BuildIndexConfig
from rag.indexing.document_loader import (
    collect_current_markdown_files,
    compute_document_content_hash,
    load_documents_from_files,
    normalize_doc_key,
)
from rag.indexing.dependency_graph import (
    DependencyCardStore,
    DocumentDependencyAggregator,
    build_dependency_vector_records,
)
from rag.indexing.embedding_client import (
    build_openai_compatible_embedder,
    embed_nodes_with_checkpoint,
    embed_texts,
)
from rag.indexing.markdown_chunker import split_documents
from rag.indexing.publication import (
    prepare_index_workspace,
    publish_index,
)
from rag.indexing.preview_renderer import write_document_chunk_previews
from rag.indexing.rebuild_decision import IndexingContext, compute_rebuild_decision
from rag.indexing.semantic_annotator import SemanticAnnotator
from rag.indexing.storage_indexer import IndexResult, MultiRouteIndexer, doc_key_from_metadata
from rag.shared.checkpoints import CheckpointStore, node_key, text_hash
from rag.shared.logging_utils import log_phase


def build_offline_index(
        docs_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        annotator: Any | None = None,
        embedder: Any | None = None,
        content_embedder: Any | None = None,
        summary_embedder: Any | None = None,
        indexer: MultiRouteIndexer | None = None,
        dependency_aggregator: Any | None = None,
        config: BuildIndexConfig | None = None,
        _managed_workspace: bool = False,
) -> IndexResult:
    """执行默认文档级增量建库流程。"""
    logger = logging.getLogger("rag.pipeline")
    cfg = config or BuildIndexConfig.from_env()
    docs_root = Path(docs_dir) if docs_dir is not None else cfg.paths.docs_dir
    storage_root = Path(storage_dir) if storage_dir is not None else cfg.paths.storage_dir

    if indexer is None:
        workspace = prepare_index_workspace(storage_root)
        logger.info(
            "使用索引构建工作区 | workspace=%s | manifest=%s",
            workspace,
            storage_root / "manifest.json",
        )
        try:
            result = build_offline_index(
                docs_dir=docs_root,
                storage_dir=workspace,
                annotator=annotator,
                embedder=embedder,
                content_embedder=content_embedder,
                summary_embedder=summary_embedder,
                indexer=MultiRouteIndexer(workspace),
                dependency_aggregator=dependency_aggregator,
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
    writer = indexer or MultiRouteIndexer(storage_root)

    # 1. 收集当前文档状态
    with log_phase(logger, "收集文档状态", docs_dir=str(docs_root)):
        current_files = collect_current_markdown_files(docs_root)
        current_hashes = {
            doc_key: compute_document_content_hash(path)
            for doc_key, path in current_files.items()
        }

    # 2. 加载历史状态
    previous_state = checkpoint.load_document_index_state()

    # 3. 构建索引上下文
    annotation_prompt = (
        cfg.annotation.prompt_path.read_text(encoding="utf-8")
        if cfg.annotation.prompt_path.exists()
        else ""
    )
    dependency_prompt = (
        cfg.dependency.prompt_path.read_text(encoding="utf-8")
        if cfg.dependency.prompt_path.exists()
        else ""
    )

    ctx = IndexingContext(
        current_doc_hashes=current_hashes,
        docs_root=docs_root,
        storage_root=storage_root,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        annotation_prompt=annotation_prompt,
        annotation_prompt_version=cfg.annotation.prompt_version,
        dependency_prompt=dependency_prompt,
        dependency_prompt_version=cfg.dependency.prompt_version,
        embedding_model=cfg.embedding.model,
        annotator_model=cfg.annotation.model,
        dependency_vector_enabled=cfg.dependency.vector_enabled,
        previous_state=previous_state,
        is_managed_workspace=_managed_workspace,
    )

    # 4. 计算重建决策
    decision = compute_rebuild_decision(ctx)

    logger.info(
        "重建决策 | 变更=%d | 删除=%d | 不变=%d | 重标注=%d | 重依赖=%d | 重向量=%d",
        len(decision.changed_docs),
        len(decision.deleted_docs),
        len(decision.unchanged_docs),
        len(decision.need_reannotate),
        len(decision.need_redependency),
        len(decision.need_reembed),
    )
    logger.info("决策原因: %s", decision.reason)

    # 5. 清理旧数据
    stale_doc_keys = decision.changed_docs | decision.deleted_docs
    stale_node_ids = gather_stale_node_ids(previous_state, stale_doc_keys)
    if stale_node_ids:
        with log_phase(logger, "清理旧索引", node_count=len(stale_node_ids)):
            writer.delete_nodes(stale_node_ids)
            if cfg.dependency.vector_enabled:
                writer.delete_dependency_nodes(stale_node_ids)
    for doc_key in stale_doc_keys:
        writer.remove_metadata_shard(doc_key)

    # 6. 无需重建的情况
    if decision.needs_no_rebuild:
        manifest = build_manifest(ctx, docs_root, storage_root)
        checkpoint.write_manifest(manifest)
        writer.rebuild_metadata_snapshot()
        writer.rebuild_bm25_from_metadata_snapshot()
        next_state = build_next_document_state(
            previous_state,
            current_hashes,
            decision.unchanged_docs,
            [],
            docs_root,
            ctx,
        )
        checkpoint.save_document_index_state(next_state)
        DependencyCardStore(storage_root).update_documents({}, decision.deleted_docs)
        checkpoint.remove_node_records(stale_node_ids - document_state_node_ids(next_state))
        checkpoint.remove_dependency_documents(decision.deleted_docs)
        logger.info("离线索引完成 | 本次无需重建，已刷新兼容索引快照")
        return IndexResult(
            node_count=0,
            content_collection="content_vec",
            summary_collection="summary_vec",
            bm25_path=writer.bm25_path,
            metadata_path=writer.metadata_path,
        )

    # 7. 仅依赖分析变化：加载已有 nodes，跳过标注和向量化
    if decision.needs_dependency_only:
        logger.info("仅需重新分析依赖 | 文档数=%d", len(decision.need_redependency))
        nodes = _read_chunks_from_checkpoint(checkpoint)
        if not nodes:
            logger.warning("未找到已标注的 chunks，无法仅重跑依赖分析")
            return IndexResult(
                node_count=0,
                content_collection="content_vec",
                summary_collection="summary_vec",
                bm25_path=writer.bm25_path,
                metadata_path=writer.metadata_path,
            )

        # 清理依赖缓存
        checkpoint.remove_dependency_documents(decision.need_redependency)

        graph_aggregator = dependency_aggregator or DocumentDependencyAggregator(
            api_key=cfg.llm.api_key or "",
            base_url=cfg.llm.base_url,
            model=cfg.dependency.model,
            prompt=dependency_prompt,
            prompt_version=cfg.dependency.prompt_version,
        )
        with log_phase(logger, "文档依赖聚合（仅依赖分析）", docs=len(decision.need_redependency), nodes=len(nodes)):
            if dependency_aggregator is None:
                dependency_result = graph_aggregator.aggregate_documents(
                    nodes,
                    current_hashes,
                    docs_root,
                    checkpoint=checkpoint,
                )
            else:
                dependency_result = graph_aggregator.aggregate_documents(
                    nodes,
                    current_hashes,
                    docs_root,
                )

        DependencyCardStore(storage_root).update_documents(
            dependency_result.documents,
            set(),
        )
        if dependency_result.failed_docs:
            raise RuntimeError(
                "依赖图聚合存在未成功文档: " + ", ".join(dependency_result.failed_docs)
            )

        manifest = build_manifest(ctx, docs_root, storage_root)
        checkpoint.write_manifest(manifest)
        next_state = build_next_document_state(
            previous_state,
            current_hashes,
            decision.unchanged_docs,
            [],
            docs_root,
            ctx,
        )
        checkpoint.save_document_index_state(next_state)

        logger.info("依赖分析完成 | 文档数=%s", len(dependency_result.documents))
        return IndexResult(
            node_count=len(nodes),
            content_collection="content_vec",
            summary_collection="summary_vec",
            bm25_path=writer.bm25_path,
            metadata_path=writer.metadata_path,
        )

    # 8. 需要重新标注：完整流程
    rebuild_keys = sorted(decision.need_reannotate)
    rebuild_files = [current_files[key] for key in rebuild_keys]

    # 先清理依赖缓存（依赖 key 基于 doc_key，不依赖 chunks）
    checkpoint.remove_dependency_documents(decision.need_redependency)

    with log_phase(logger, "加载文档", docs=len(rebuild_files)):
        documents = load_documents_from_files(rebuild_files, docs_root)
    if not documents:
        raise RuntimeError("存在需要重建的文档，但未加载到任何文档")
    logger.info("文档加载完成 | 文档数=%s", len(documents))

    with log_phase(
        logger,
        "切分 chunk",
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    ):
        nodes = split_documents(documents, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    logger.info("切分完成 | chunk数=%s", len(nodes))

    # 切分后再清理标注缓存（基于新生成的 node_id）
    checkpoint.remove_annotation_by_node_ids({node_key(node) for node in nodes})

    manifest = build_manifest(ctx, docs_root, storage_root)
    checkpoint.write_manifest(manifest)
    checkpoint.write_chunks(nodes)
    write_document_chunk_previews(nodes, storage_root)

    semantic_annotator = annotator or SemanticAnnotator(
        api_key=cfg.llm.api_key or "",
        base_url=cfg.llm.base_url,
        model=cfg.annotation.model,
        max_workers=cfg.annotation.workers,
        prompt=annotation_prompt,
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

    graph_aggregator = dependency_aggregator or DocumentDependencyAggregator(
        api_key=cfg.llm.api_key or "",
        base_url=cfg.llm.base_url,
        model=cfg.dependency.model,
        prompt=dependency_prompt,
        prompt_version=cfg.dependency.prompt_version,
    )
    with log_phase(logger, "文档依赖聚合", docs=len(rebuild_keys), nodes=len(nodes)):
        if dependency_aggregator is None:
            dependency_result = graph_aggregator.aggregate_documents(
                nodes,
                current_hashes,
                docs_root,
                checkpoint=checkpoint,
            )
        else:
            dependency_result = graph_aggregator.aggregate_documents(
                nodes,
                current_hashes,
                docs_root,
            )
    DependencyCardStore(storage_root).update_documents(
        dependency_result.documents,
        decision.deleted_docs,
    )
    if dependency_result.failed_docs:
        raise RuntimeError(
            "依赖图聚合存在未成功文档，不发布当前索引批次: "
            + ", ".join(dependency_result.failed_docs)
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

    dependency_vector_records: list[dict[str, Any]] = []
    dependency_embeddings: list[list[float]] = []
    if cfg.dependency.vector_enabled:
        dependency_vector_records = build_dependency_vector_records(
            nodes,
            dependency_result.documents,
            docs_root,
        )
        with log_phase(logger, "dependency 向量化", records=len(dependency_vector_records)):
            dependency_embeddings = embed_texts(
                content_model,
                [record["text"] for record in dependency_vector_records],
            )

    with log_phase(logger, "多路入库", nodes=len(nodes), storage_dir=str(storage_root)):
        result = writer.index(nodes, content_embeddings, summary_embeddings, root_doc_dir=docs_root)
        if cfg.dependency.vector_enabled:
            writer.index_dependency_vectors(dependency_vector_records, dependency_embeddings)
    next_state = build_next_document_state(
        previous_state,
        current_hashes,
        decision.unchanged_docs,
        nodes,
        docs_root,
        ctx,
    )
    checkpoint.save_document_index_state(next_state)
    checkpoint.remove_node_records(stale_node_ids - document_state_node_ids(next_state))
    checkpoint.remove_dependency_documents(decision.deleted_docs)
    logger.info(
        "离线索引完成 | chunk数=%s | metadata=%s | bm25=%s",
        result.node_count,
        result.metadata_path,
        result.bm25_path,
    )
    return result


def _read_chunks_from_checkpoint(checkpoint: CheckpointStore) -> list[Any]:
    """从检查点重建 node 对象列表。"""
    from llama_index.core.schema import TextNode

    records = checkpoint.load_chunk_records()
    nodes: list[Any] = []
    for record in records.values():
        node = TextNode(
            id_=record.get("node_id", ""),
            text=record.get("text", ""),
            metadata=record.get("metadata", {}),
        )
        nodes.append(node)
    return nodes


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
    ctx: IndexingContext,
) -> dict[str, Any]:
    """构建下次索引的文档状态快照。"""
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
            "node_count": len(rebuilt_node_ids.get(doc_key, [])),
            "updated_at": updated_at,
        }

    annotation_prompt_hash = text_hash(ctx.annotation_prompt) if ctx.annotation_prompt else ""
    dependency_prompt_hash = text_hash(ctx.dependency_prompt) if ctx.dependency_prompt else ""

    return {
        "schema_version": "v1",
        "chunk_size": ctx.chunk_size,
        "chunk_overlap": ctx.chunk_overlap,
        "annotation_prompt_version": ctx.annotation_prompt_version,
        "annotation_prompt_hash": annotation_prompt_hash,
        "dependency_prompt_version": ctx.dependency_prompt_version,
        "dependency_prompt_hash": dependency_prompt_hash,
        "embedding_model": ctx.embedding_model,
        "total_docs": len(docs),
        "total_nodes": sum(len(record.get("node_ids", [])) for record in docs.values()),
        "docs": docs,
    }


def document_state_node_ids(state: dict[str, Any]) -> set[str]:
    docs = state.get("docs")
    if not isinstance(docs, dict):
        return set()
    return {
        str(node_id)
        for record in docs.values()
        if isinstance(record, dict)
        for node_id in record.get("node_ids", [])
        if node_id
    }


def group_node_ids_by_doc_key(nodes: list[Any], docs_root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_metadata(metadata, docs_root)
        node_id = str(getattr(node, "node_id", None) or getattr(node, "id_", None) or "")
        if doc_key and node_id:
            groups.setdefault(doc_key, []).append(node_id)
    return groups


def build_manifest(ctx: IndexingContext, docs_root: Path, storage_root: Path) -> dict[str, Any]:
    """构建 manifest.json 配置清单。"""
    return {
        "docs_dir": str(docs_root),
        "storage_dir": str(storage_root),
        "chunk_size": ctx.chunk_size,
        "chunk_overlap": ctx.chunk_overlap,
        "annotator_model": ctx.annotator_model,
        "annotation_prompt_version": ctx.annotation_prompt_version,
        "annotation_prompt_hash": text_hash(ctx.annotation_prompt) if ctx.annotation_prompt else "",
        "dependency_prompt_version": ctx.dependency_prompt_version,
        "dependency_prompt_hash": text_hash(ctx.dependency_prompt) if ctx.dependency_prompt else "",
        "embedding_model": ctx.embedding_model,
        "dependency_vector_enabled": ctx.dependency_vector_enabled,
    }

