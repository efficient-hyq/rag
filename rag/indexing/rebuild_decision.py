"""统一的重建决策逻辑模块。

该模块负责判断在各种情况下需要重建哪些阶段（切分、标注、依赖分析、向量化）。
通过比对当前配置与历史状态，计算出最小化的重建范围。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.indexing.dependency_graph import DEPENDENCY_SCHEMA_VERSION
from rag.shared.checkpoints import text_hash


@dataclass
class RebuildDecision:
    """重建决策结果。

    Attributes:
        changed_docs: 内容变化的文档 key 集合（新增或修改）
        deleted_docs: 删除的文档 key 集合
        unchanged_docs: 内容未变化的文档 key 集合
        need_rechunk: 是否需要重新切分（chunk_size/overlap 变化）
        need_reannotate: 需要重新标注的文档 key 集合
        need_redependency: 需要重新分析依赖的文档 key 集合
        need_reembed: 需要重新向量化的文档 key 集合
        reason: 决策原因说明
    """
    changed_docs: set[str]
    deleted_docs: set[str]
    unchanged_docs: set[str]
    need_rechunk: bool
    need_reannotate: set[str]
    need_redependency: set[str]
    need_reembed: set[str]
    reason: str

    @property
    def needs_dependency_only(self) -> bool:
        """是否仅需重建依赖（标注不变）。"""
        return bool(self.need_redependency) and not self.need_reannotate

    @property
    def needs_no_rebuild(self) -> bool:
        """是否完全无需重建。"""
        return not self.need_reannotate and not self.need_redependency


@dataclass
class IndexingContext:
    """索引构建上下文信息。"""
    # 当前文档状态
    current_doc_hashes: dict[str, str]
    docs_root: Path
    storage_root: Path

    # 当前配置
    chunk_size: int
    chunk_overlap: int
    annotation_prompt: str
    annotation_prompt_version: str
    dependency_prompt: str
    dependency_prompt_version: str
    embedding_model: str
    annotator_model: str
    dependency_vector_enabled: bool

    # 历史状态
    previous_state: dict[str, Any]

    # 特殊标志
    is_managed_workspace: bool = False


def compute_rebuild_decision(ctx: IndexingContext) -> RebuildDecision:
    """计算重建决策。

    统一入口，根据当前上下文判断需要重建哪些阶段。
    决策逻辑：
    1. 文档内容变化 → 该文档需要重新标注、依赖、向量化
    2. chunk 配置变化 → 所有文档需要重新切分、标注、依赖、向量化
    3. 标注协议/prompt 变化 → 所有文档需要重新标注、依赖、向量化
    4. 依赖协议/prompt 变化 → 所有文档需要重新分析依赖
    5. 向量化模型变化 → 所有文档需要重新向量化
    """
    # 1. 计算文档内容变化
    previous_docs = ctx.previous_state.get("docs", {})
    if not isinstance(previous_docs, dict):
        previous_docs = {}

    changed_docs = set()
    deleted_docs = set()
    unchanged_docs = set()

    for doc_key, current_hash in ctx.current_doc_hashes.items():
        previous_record = previous_docs.get(doc_key)
        if not isinstance(previous_record, dict):
            changed_docs.add(doc_key)
        elif previous_record.get("content_hash") != current_hash:
            changed_docs.add(doc_key)
        else:
            unchanged_docs.add(doc_key)

    for doc_key in previous_docs:
        if doc_key not in ctx.current_doc_hashes:
            deleted_docs.add(doc_key)

    # 2. 检查各类配置变化
    chunk_config_changed = _check_chunk_config_changed(ctx)
    annotation_changed = _check_annotation_changed(ctx)
    dependency_changed = _check_dependency_changed(ctx)
    embedding_model_changed = _check_embedding_model_changed(ctx)
    dependency_schema_changed = _check_dependency_schema_changed(ctx.storage_root, ctx.current_doc_hashes)

    # 3. 决策逻辑
    reasons = []

    # 需要重新切分
    need_rechunk = chunk_config_changed
    if chunk_config_changed:
        reasons.append(f"chunk 配置变化: size={ctx.chunk_size}, overlap={ctx.chunk_overlap}")

    # 需要重新标注的文档
    need_reannotate = set(changed_docs)
    if chunk_config_changed or annotation_changed:
        need_reannotate.update(unchanged_docs)
        if annotation_changed:
            reasons.append(annotation_changed)

    # 需要重新分析依赖的文档
    need_redependency = set(changed_docs)
    if chunk_config_changed or annotation_changed:
        need_redependency.update(unchanged_docs)
    if dependency_changed:
        need_redependency.update(unchanged_docs)
        reasons.append(dependency_changed)
    if dependency_schema_changed and ctx.is_managed_workspace:
        need_redependency.update(unchanged_docs)
        reasons.append("依赖 schema 版本变化")

    # 需要重新向量化的文档
    need_reembed = set(changed_docs)
    if chunk_config_changed or annotation_changed:
        need_reembed.update(unchanged_docs)
    if embedding_model_changed:
        need_reembed.update(unchanged_docs)
        reasons.append(f"向量化模型变化: {ctx.previous_state.get('embedding_model', '')} → {ctx.embedding_model}")

    if changed_docs:
        reasons.insert(0, f"文档内容变化: 新增/修改={len(changed_docs)}, 删除={len(deleted_docs)}")

    if not reasons:
        reasons.append("无变化")

    return RebuildDecision(
        changed_docs=changed_docs,
        deleted_docs=deleted_docs,
        unchanged_docs=unchanged_docs,
        need_rechunk=need_rechunk,
        need_reannotate=need_reannotate,
        need_redependency=need_redependency,
        need_reembed=need_reembed,
        reason=" | ".join(reasons),
    )


def _check_chunk_config_changed(ctx: IndexingContext) -> bool:
    """检查 chunk 配置是否变化。"""
    prev_chunk_size = ctx.previous_state.get("chunk_size")
    prev_chunk_overlap = ctx.previous_state.get("chunk_overlap")
    return prev_chunk_size != ctx.chunk_size or prev_chunk_overlap != ctx.chunk_overlap


def _check_annotation_changed(ctx: IndexingContext) -> str | None:
    """检查标注配置是否变化，返回变化原因或 None。"""
    annotation_prompt_hash = text_hash(ctx.annotation_prompt) if ctx.annotation_prompt else ""
    prev_annotation_hash = ctx.previous_state.get("annotation_prompt_hash", "")
    prev_annotation_version = ctx.previous_state.get("annotation_prompt_version", "")

    if prev_annotation_version and ctx.annotation_prompt_version != prev_annotation_version:
        return f"标注协议升级: {prev_annotation_version} → {ctx.annotation_prompt_version}"
    if annotation_prompt_hash and annotation_prompt_hash != prev_annotation_hash:
        return f"标注 prompt 内容变化: {prev_annotation_hash[:8]} → {annotation_prompt_hash[:8]}"
    return None


def _check_dependency_changed(ctx: IndexingContext) -> str | None:
    """检查依赖配置是否变化，返回变化原因或 None。"""
    dependency_prompt_hash = text_hash(ctx.dependency_prompt) if ctx.dependency_prompt else ""
    prev_dependency_hash = ctx.previous_state.get("dependency_prompt_hash", "")
    prev_dependency_version = ctx.previous_state.get("dependency_prompt_version", "")

    if prev_dependency_version and ctx.dependency_prompt_version != prev_dependency_version:
        return f"依赖协议升级: {prev_dependency_version} → {ctx.dependency_prompt_version}"
    if dependency_prompt_hash and dependency_prompt_hash != prev_dependency_hash:
        return f"依赖 prompt 内容变化: {prev_dependency_hash[:8]} → {dependency_prompt_hash[:8]}"
    return None


def _check_embedding_model_changed(ctx: IndexingContext) -> bool:
    """检查向量化模型是否变化。"""
    prev_embedding_model = ctx.previous_state.get("embedding_model", "")
    return bool(prev_embedding_model and ctx.embedding_model != prev_embedding_model)


def _check_dependency_schema_changed(
    storage_root: Path,
    current_hashes: dict[str, str],
) -> bool:
    """检查依赖 schema 版本是否变化，或者依赖卡片数据是否不完整。"""
    path = storage_root / "dependency_cards.json"
    if not path.exists():
        return bool(current_hashes)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(payload, dict):
        return True
    documents = payload.get("documents")
    if payload.get("schema_version") != DEPENDENCY_SCHEMA_VERSION or not isinstance(documents, dict):
        return True
    # 检查是否所有文档都有完整的依赖卡片
    for doc_key, content_hash in current_hashes.items():
        card = documents.get(doc_key)
        if (
            not isinstance(card, dict)
            or card.get("status") != "success"
            or card.get("doc_content_hash") != content_hash
        ):
            return True
    return False
