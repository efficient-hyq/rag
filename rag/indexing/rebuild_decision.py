"""按已切分 chunk 判断替换、删除和需要重跑的阶段。

建库不再扫描 Markdown 原文。当前 chunk 集合和上次入库的 chunk 指纹直接比较：
新增或正文变化的 chunk 重新标注并嵌入，消失的 chunk 从索引删除。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.indexing.chunk_context import CONTEXT_VERSION
from rag.indexing.dependency_graph import DEPENDENCY_SCHEMA_VERSION
from rag.shared.checkpoints import text_hash


CHUNK_STATE_SCHEMA = "chunk_index_v2"


@dataclass
class ChunkRebuildDecision:
    """一次建库要处理的 chunk 和文档依赖范围。"""

    added: set[str]
    replaced: set[str]
    deleted: set[str]
    unchanged: set[str]
    need_reannotate: set[str]
    need_reembed: set[str]
    need_redependency_docs: set[str]
    deleted_docs: set[str]
    reason: str

    @property
    def needs_no_rebuild(self) -> bool:
        return (
            not self.added
            and not self.replaced
            and not self.deleted
            and not self.deleted_docs
            and not self.need_reannotate
            and not self.need_reembed
            and not self.need_redependency_docs
        )


def compute_chunk_rebuild_decision(
    current_hashes: dict[str, str],
    current_doc_hashes: dict[str, str],
    chunk_docs: dict[str, str],
    previous_state: dict[str, Any],
    *,
    annotation_prompt_version: str,
    annotation_prompt_hash: str,
    dependency_prompt_version: str,
    dependency_prompt_hash: str,
    embedding_model: str,
    storage_root: Path,
    situating_prompt_version: str = "",
    situating_prompt_hash: str = "",
    situating_doc_hashes: dict[str, str] | None = None,
) -> ChunkRebuildDecision:
    """比较当前 chunk 与上次状态，得到替换、删除和重跑范围。"""
    previous_chunks = _previous_chunk_hashes(previous_state)
    previous_docs = _previous_doc_keys(previous_state)
    added = set(current_hashes) - set(previous_chunks)
    deleted = set(previous_chunks) - set(current_hashes)
    replaced = {
        node_id
        for node_id, content_hash in current_hashes.items()
        if node_id in previous_chunks and previous_chunks[node_id] != content_hash
    }
    unchanged = set(current_hashes) - added - replaced

    reasons: list[str] = []
    annotation_changed = _prompt_changed(
        previous_state.get("annotation_prompt_version", ""),
        previous_state.get("annotation_prompt_hash", ""),
        annotation_prompt_version,
        annotation_prompt_hash,
    )
    dependency_changed = _prompt_changed(
        previous_state.get("dependency_prompt_version", ""),
        previous_state.get("dependency_prompt_hash", ""),
        dependency_prompt_version,
        dependency_prompt_hash,
    )
    context_changed = (
        previous_state.get("schema_version") == CHUNK_STATE_SCHEMA
        and previous_state.get("context_version") != CONTEXT_VERSION
    )
    previous_model = str(previous_state.get("embedding_model") or "")
    embedding_model_changed = bool(previous_model and previous_model != embedding_model)
    schema_changed = _dependency_schema_changed(storage_root, current_doc_hashes)

    need_reannotate = set(added | replaced)
    if annotation_changed or context_changed:
        need_reannotate.update(current_hashes)
        if annotation_changed:
            reasons.append(annotation_changed)
        if context_changed:
            reasons.append(f"上下文版本变化: {previous_state.get('context_version') or ''} → {CONTEXT_VERSION}")

    situating_changed = _prompt_changed(
        previous_state.get("situating_prompt_version", ""),
        previous_state.get("situating_prompt_hash", ""),
        situating_prompt_version,
        situating_prompt_hash,
    )
    if (
        not situating_changed
        and situating_prompt_hash
        and previous_state.get("schema_version") == CHUNK_STATE_SCHEMA
        and not str(previous_state.get("situating_prompt_hash") or "")
    ):
        situating_changed = "新增片段定位上下文"
    if situating_changed:
        need_reannotate.update(current_hashes)
        reasons.append(situating_changed)

    previous_doc_texts = previous_state.get("situating_doc_hashes")
    if isinstance(previous_doc_texts, dict) and situating_doc_hashes:
        document_changed = False
        for node_id, doc_key in chunk_docs.items():
            previous_hash = str(previous_doc_texts.get(doc_key) or "")
            current_hash = str(situating_doc_hashes.get(doc_key) or "")
            if previous_hash and current_hash and previous_hash != current_hash:
                need_reannotate.add(node_id)
                document_changed = True
        if document_changed:
            reasons.append("整篇文档变化，需要重写定位上下文和标注")

    need_reembed = set(need_reannotate)
    if embedding_model_changed:
        need_reembed.update(current_hashes)
        reasons.append(f"向量化模型变化: {previous_model} → {embedding_model}")

    need_docs: set[str] = set()
    for node_id in need_reannotate | replaced | added:
        doc_key = chunk_docs.get(node_id)
        if doc_key and doc_key in current_doc_hashes:
            need_docs.add(doc_key)
    previous_doc_hashes = _previous_doc_hashes(previous_state)
    for doc_key, doc_hash in current_doc_hashes.items():
        if previous_doc_hashes.get(doc_key) != doc_hash:
            need_docs.add(doc_key)
    if dependency_changed or schema_changed:
        need_docs.update(current_doc_hashes)
        if dependency_changed:
            reasons.append(dependency_changed)
        if schema_changed:
            reasons.append("依赖卡片缺失或与当前 chunk 不一致")

    deleted_docs = previous_docs - set(current_doc_hashes)
    if added or replaced or deleted:
        reasons.insert(
            0,
            f"chunk 变化: 新增={len(added)}, 替换={len(replaced)}, 删除={len(deleted)}",
        )
    if not reasons:
        reasons.append("无变化")

    return ChunkRebuildDecision(
        added=added,
        replaced=replaced,
        deleted=deleted,
        unchanged=unchanged,
        need_reannotate=need_reannotate,
        need_reembed=need_reembed,
        need_redependency_docs=need_docs,
        deleted_docs=deleted_docs,
        reason=" | ".join(reasons),
    )


def previous_node_ids(state: dict[str, Any]) -> set[str]:
    """上次入库的 chunk。新状态读 chunks，旧的文档状态则展开其中的 node_id。"""
    if state.get("schema_version") == CHUNK_STATE_SCHEMA and isinstance(state.get("chunks"), dict):
        return {str(node_id) for node_id in state["chunks"]}
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


def _previous_chunk_hashes(state: dict[str, Any]) -> dict[str, str]:
    if state.get("schema_version") == CHUNK_STATE_SCHEMA and isinstance(state.get("chunks"), dict):
        hashes: dict[str, str] = {}
        for node_id, record in state["chunks"].items():
            if isinstance(record, dict) and record.get("content_hash"):
                hashes[str(node_id)] = str(record["content_hash"])
        return hashes
    # 旧状态没有 chunk 指纹。把当时的 node 视为待删除，当前 chunk 全部按新增处理。
    return {node_id: "" for node_id in previous_node_ids(state)}


def _previous_doc_keys(state: dict[str, Any]) -> set[str]:
    docs = state.get("docs")
    if not isinstance(docs, dict):
        return set()
    return {str(doc_key) for doc_key in docs}


def _previous_doc_hashes(state: dict[str, Any]) -> dict[str, str]:
    docs = state.get("docs")
    if not isinstance(docs, dict):
        return {}
    hashes: dict[str, str] = {}
    for doc_key, record in docs.items():
        if isinstance(record, dict) and record.get("content_hash"):
            hashes[str(doc_key)] = str(record["content_hash"])
    return hashes


def _prompt_changed(
    previous_version: object,
    previous_hash: object,
    version: str,
    prompt_hash: str,
) -> str | None:
    prev_version = str(previous_version or "")
    prev_hash = str(previous_hash or "")
    if not prev_version and not prev_hash:
        return None
    if prev_version and version != prev_version:
        return f"协议升级: {prev_version} → {version}"
    if prev_hash and prompt_hash and prompt_hash != prev_hash:
        return f"prompt 内容变化: {prev_hash[:8]} → {prompt_hash[:8]}"
    return None


def _dependency_schema_changed(storage_root: Path, current_hashes: dict[str, str]) -> bool:
    """依赖卡片必须覆盖当前文档，且指纹与这次的 chunk 集合一致。"""
    path = Path(storage_root) / "dependency_cards.json"
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
    if set(documents) != set(current_hashes):
        return True
    for doc_key, content_hash in current_hashes.items():
        card = documents.get(doc_key)
        if (
            not isinstance(card, dict)
            or card.get("status") != "success"
            or card.get("doc_content_hash") != content_hash
        ):
            return True
    return False


def prompt_fingerprint(prompt: str) -> str:
    return text_hash(prompt) if prompt else ""
