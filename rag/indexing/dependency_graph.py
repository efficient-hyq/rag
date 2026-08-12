"""文档级隐式依赖图聚合、校验与持久化。"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag.indexing.dependency_topics import normalize_relation_edge
from rag.indexing.storage_indexer import doc_key_from_metadata
from rag.shared.checkpoints import CheckpointStore, ProgressSnapshot, node_key, print_progress, text_hash


DEPENDENCY_SCHEMA_VERSION = "dependency_graph_v4"
DEFAULT_DEPENDENCY_PROMPT_VERSION = "dependency_graph_v4"


@dataclass(frozen=True)
class DependencyAggregationResult:
    documents: dict[str, dict[str, Any]]
    failed_docs: list[str]

    @property
    def edge_count(self) -> int:
        return sum(
            len(edges)
            for document in self.documents.values()
            for edges in document.get("node_dependencies", {}).values()
        )


class DocumentDependencyAggregator:
    """按单个 Markdown 文档调用 LLM，生成一层直接依赖图。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.6-plus",
        max_retries: int = 3,
        client: OpenAI | None = None,
        prompt: str | None = None,
        prompt_version: str = DEFAULT_DEPENDENCY_PROMPT_VERSION,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("缺少依赖图聚合所需的 LLM API Key")
        self.model = model
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt or load_dependency_prompt()
        self.prompt_version = prompt_version
        self.prompt_hash = text_hash(self.prompt)

    def aggregate_documents(
        self,
        nodes: list[Any],
        doc_hashes: dict[str, str],
        root_doc_dir: str | Path,
        checkpoint: CheckpointStore | None = None,
        show_progress: bool = True,
    ) -> DependencyAggregationResult:
        logger = logging.getLogger("rag.dependency_graph")
        groups = group_nodes_by_doc(nodes, Path(root_doc_dir))
        cached = checkpoint.load_dependency_cards() if checkpoint else {}
        documents: dict[str, dict[str, Any]] = {}
        failed_docs: list[str] = []
        completed = 0
        if show_progress:
            print_progress(ProgressSnapshot("依赖聚合", len(groups), 0, 0))

        for doc_key, doc_nodes in sorted(groups.items()):
            cache_key = self._cache_key(doc_key, doc_hashes.get(doc_key, ""))
            cached_card = cached.get(cache_key)
            if cached_card and cached_card.get("status") == "success":
                card = dict(cached_card)
            else:
                card = self._aggregate_document(doc_key, doc_nodes, doc_hashes.get(doc_key, ""))
                if checkpoint:
                    checkpoint.append_dependency_card(cache_key, card)
            documents[doc_key] = card
            if card.get("status") != "success":
                failed_docs.append(doc_key)
            completed += 1
            if show_progress:
                print_progress(ProgressSnapshot("依赖聚合", len(groups), completed, len(failed_docs)))

        logger.info(
            "依赖聚合结束 | 文档数=%s | 成功=%s | 失败=%s | 依赖边=%s",
            len(documents),
            len(documents) - len(failed_docs),
            len(failed_docs),
            DependencyAggregationResult(documents, failed_docs).edge_count,
        )
        return DependencyAggregationResult(documents=documents, failed_docs=failed_docs)

    def _aggregate_document(
        self,
        doc_key: str,
        nodes: list[Any],
        doc_content_hash: str,
    ) -> dict[str, Any]:
        payload = build_aggregation_input(nodes)
        node_ids = [item["node_id"] for item in payload]
        if not node_ids or any(not item["text"].strip() for item in payload):
            return build_failed_document_card(
                doc_content_hash,
                len(node_ids),
                self.model,
                self.prompt_version,
                "聚合输入缺少 node 或完整 chunk text",
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"doc_id": doc_key, "nodes": payload},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                raw = parse_json_object(content)
                return validate_dependency_document(
                    raw,
                    node_ids,
                    doc_content_hash,
                    self.model,
                    self.prompt_version,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        return build_failed_document_card(
            doc_content_hash,
            len(node_ids),
            self.model,
            self.prompt_version,
            str(last_error or "依赖聚合失败"),
        )

    def _cache_key(self, doc_key: str, doc_hash: str) -> str:
        return f"{doc_key}|{doc_hash}|{self.prompt_version}|{self.prompt_hash}"


class DependencyCardStore:
    """维护构建工作区中的增量依赖卡片。"""

    def __init__(self, storage_dir: str | Path) -> None:
        self.path = Path(storage_dir) / "dependency_cards.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": DEPENDENCY_SCHEMA_VERSION,
                "documents": {},
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("dependency_cards.json 顶层必须是对象")
        documents = payload.get("documents")
        return {
            "schema_version": DEPENDENCY_SCHEMA_VERSION,
            "documents": documents if isinstance(documents, dict) else {},
        }

    def update_documents(
        self,
        rebuilt_documents: dict[str, dict[str, Any]],
        removed_doc_keys: set[str],
    ) -> dict[str, Any]:
        payload = self.load()
        documents = dict(payload.get("documents") or {})
        for doc_key in removed_doc_keys:
            documents.pop(doc_key, None)
        documents.update(rebuilt_documents)
        payload["documents"] = {key: documents[key] for key in sorted(documents)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def build_aggregation_input(nodes: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: int((getattr(item, "metadata", {}) or {}).get("chunk_index", 0))):
        metadata = dict(getattr(node, "metadata", {}) or {})
        result.append(
            {
                "node_id": node_key(node),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "heading_path": str(metadata.get("heading_path") or ""),
                "summary": str(metadata.get("summary") or ""),
                "keywords": list(metadata.get("keywords") or []),
                "dependency_topics": list(metadata.get("dependency_topics") or []),
                "text": str(getattr(node, "text", "")),
            }
        )
    return result


def group_nodes_by_doc(nodes: list[Any], root_doc_dir: Path) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_metadata(metadata, root_doc_dir)
        groups.setdefault(doc_key, []).append(node)
    return groups


def build_dependency_vector_records(
    nodes: list[Any],
    documents: dict[str, dict[str, Any]],
    root_doc_dir: str | Path,
) -> list[dict[str, Any]]:
    """以存在有效依赖的 source node 为单位生成向量文本。"""
    node_by_id = {node_key(node): node for node in nodes}
    doc_groups = group_nodes_by_doc(nodes, Path(root_doc_dir))
    records: list[dict[str, Any]] = []
    for doc_key, document in documents.items():
        if document.get("status") != "success":
            continue
        document_node_ids = {node_key(node) for node in doc_groups.get(doc_key, [])}
        dependencies = document.get("node_dependencies", {})
        for source_id, edges in dependencies.items():
            valid_edges = [edge for edge in edges if isinstance(edge, dict)] if isinstance(edges, list) else []
            if not valid_edges or source_id not in document_node_ids or source_id not in node_by_id:
                continue
            source = node_by_id[source_id]
            source_metadata = dict(getattr(source, "metadata", {}) or {})
            target_summaries = [
                str((getattr(node_by_id.get(str(edge.get("node_id"))), "metadata", {}) or {}).get("summary") or "")
                for edge in valid_edges
            ]
            topics = [str(edge.get("topic") or "") for edge in valid_edges]
            reasons = [str(edge.get("reason") or "") for edge in valid_edges]
            vector_text = "\n".join(
                [
                    f"当前内容：{source_metadata.get('summary') or getattr(source, 'text', '')}",
                    f"章节：{source_metadata.get('heading_path') or ''}",
                    f"依赖主题：{'、'.join(item for item in topics if item)}",
                    "相关依赖内容：" + "；".join(
                        item for item in target_summaries + reasons if item
                    ),
                ]
            )
            records.append(
                {
                    "node_id": source_id,
                    "text": vector_text,
                    "metadata": {
                        "source_node_id": source_id,
                        "doc_id": str(source_metadata.get("doc_id") or doc_key),
                        "dependency_node_ids": [str(edge.get("node_id") or "") for edge in valid_edges],
                        "dependency_topics": [
                            str(edge.get("topic") or "")
                            for edge in valid_edges
                            if edge.get("topic")
                        ],
                    },
                }
            )
    return records


def validate_dependency_document(
    raw: dict[str, Any],
    node_ids: list[str],
    doc_content_hash: str,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    logger = logging.getLogger("rag.dependency_graph")
    node_id_set = set(node_ids)
    raw_dependencies = raw.get("node_dependencies") if isinstance(raw, dict) else None
    if not isinstance(raw_dependencies, dict):
        return build_failed_document_card(
            doc_content_hash,
            len(node_ids),
            model,
            prompt_version,
            "LLM 输出缺少 node_dependencies 对象",
        )

    processed_sources = node_id_set.intersection(str(key) for key in raw_dependencies)
    dependencies: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    discarded = 0
    for source_id, raw_edges in raw_dependencies.items():
        source_id = str(source_id)
        if source_id not in node_id_set or not isinstance(raw_edges, list):
            discarded += 1
            continue
        seen_targets: set[str] = set()
        for raw_edge in raw_edges:
            edge = normalize_relation_edge(raw_edge)
            if (
                edge is None
                or edge["node_id"] not in node_id_set
                or edge["node_id"] == source_id
                or edge["node_id"] in seen_targets
            ):
                discarded += 1
                continue
            dependencies[source_id].append(edge)
            seen_targets.add(edge["node_id"])

    status = "success" if len(processed_sources) == len(node_ids) else "partial"
    error = None if status == "success" else "node_dependencies 未覆盖文档全部 node"
    if discarded:
        logger.warning("依赖边校验丢弃 | 数量=%s", discarded)
    return {
        "status": status,
        "doc_content_hash": doc_content_hash,
        "expected_node_count": len(node_ids),
        "processed_node_count": len(processed_sources),
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": utc_now(),
        "error": error,
        "discarded_edge_count": discarded,
        "node_dependencies": dependencies,
    }


def build_failed_document_card(
    doc_content_hash: str,
    expected_node_count: int,
    model: str,
    prompt_version: str,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "doc_content_hash": doc_content_hash,
        "expected_node_count": expected_node_count,
        "processed_node_count": 0,
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": utc_now(),
        "error": error,
        "discarded_edge_count": 0,
        "node_dependencies": {},
    }


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("依赖聚合输出必须是 JSON 对象")
    return payload


def load_dependency_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else Path("prompts/dependency_graph_v4.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_DEPENDENCY_PROMPT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


DEFAULT_DEPENDENCY_PROMPT = """
你是文档内部直接依赖关系聚合器。输入只包含一个 Markdown 文档的全部 chunk。
只依据 chunk 完整原文建立直接依赖，无法确认时输出空数组，不使用通用常识补充。
依赖双方必须是输入中的 node_id，不允许自身依赖、重复边或跨文档关系。
relation_type 只能取 prerequisite、constraint、failure_handling、follow_up。
每条边必须包含 node_id、relation_type、required、0 到 1 的 confidence 和非空 reason。
topic 是可选的自由文本说明；不得因为 topic 为空或与已有标注不同而丢弃依赖边。
follow_up 通常 required=false。只返回严格 JSON：{"node_dependencies": {"node-id": []}}。
每个输入 node_id 都必须作为 node_dependencies 的 key 出现，无依赖时值为 []。
""".strip()
