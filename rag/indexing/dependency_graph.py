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
DEFAULT_DEPENDENCY_PROMPT_VERSION = "dependency_graph_v5"


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
        notes = build_dependency_notes(nodes)
        node_ids = [item["node_id"] for item in notes]
        if not node_ids or any(not item for item in node_ids):
            return build_failed_document_card(
                doc_content_hash,
                len(node_ids),
                self.model,
                self.prompt_version,
                "聚合输入缺少 node",
            )

        batches = iter_note_batches(notes)
        logger = logging.getLogger("rag.dependency_graph")
        if not batches:
            logger.info("依赖整理跳过模型 | 文档=%s | 段落=%s | 原因=没有规矩", doc_key, len(notes))
            return build_note_document_card(
                [],
                node_ids,
                doc_content_hash,
                self.model,
                self.prompt_version,
            )

        collected: list[list[tuple[str, dict[str, Any]]]] = []
        last_error: Exception | None = None
        for index, batch in enumerate(batches, start=1):
            edges, error = self._request_note_edges(doc_key, batch)
            if error is not None:
                last_error = error
                logger.warning(
                    "依赖整理失败 | 文档=%s | 批次=%s/%s | error=%s",
                    doc_key,
                    index,
                    len(batches),
                    error,
                )
                break
            collected.append(edges)
        if last_error is not None:
            return build_failed_document_card(
                doc_content_hash,
                len(node_ids),
                self.model,
                self.prompt_version,
                str(last_error),
            )
        logger.info("依赖整理完成 | 文档=%s | 段落=%s | 请求=%s", doc_key, len(notes), len(batches))
        return build_note_document_card(
            merge_note_edges(collected),
            node_ids,
            doc_content_hash,
            self.model,
            self.prompt_version,
        )

    def _request_note_edges(
        self,
        doc_key: str,
        batch: dict[str, Any],
    ) -> tuple[list[tuple[str, dict[str, Any]]], Exception | None]:
        payload = {"doc_id": doc_key, **batch}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                return extract_note_edges(parse_json_object(content)), None
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        return [], last_error or RuntimeError("依赖整理失败")

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


def build_dependency_notes(nodes: list[Any]) -> list[dict[str, Any]]:
    """把标注好的身份和规矩收成依赖整理输入，不带正文全文。"""
    ordered = sorted(nodes, key=lambda item: int((getattr(item, "metadata", {}) or {}).get("chunk_index", 0)))
    notes: list[dict[str, Any]] = []
    for node in ordered:
        metadata = dict(getattr(node, "metadata", {}) or {})
        subject = metadata.get("subject") if isinstance(metadata.get("subject"), dict) else {}
        rules = metadata.get("stated_rules") if isinstance(metadata.get("stated_rules"), list) else []
        notes.append(
            {
                "node_id": node_key(node),
                "heading_path": str(metadata.get("heading_path") or ""),
                "subject": {
                    "kind": str(subject.get("kind") or "其他"),
                    "title": str(subject.get("title") or ""),
                    "identifiers": [str(item) for item in subject.get("identifiers") or [] if str(item).strip()],
                },
                "stated_rules": [dict(rule) for rule in rules if isinstance(rule, dict)],
            }
        )
    return notes


def iter_note_batches(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """有规矩时一次送出整篇文档的笔记。没有规矩时不请求模型。"""
    has_rule = any(
        isinstance(rule, dict)
        for note in notes
        for rule in note.get("stated_rules") or []
    )
    if not has_rule:
        return []
    return [{"chunks": notes}]


def extract_note_edges(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    dependencies = raw.get("node_dependencies") if isinstance(raw, dict) else None
    if not isinstance(dependencies, dict):
        raise ValueError("LLM 输出缺少 node_dependencies 对象")
    edges: list[tuple[str, dict[str, Any]]] = []
    for source_id, raw_edges in dependencies.items():
        if not isinstance(raw_edges, list):
            continue
        for raw_edge in raw_edges:
            edge = normalize_relation_edge(raw_edge)
            if edge is not None:
                edges.append((str(source_id), edge))
    return edges


def merge_note_edges(
    groups: list[list[tuple[str, dict[str, Any]]]],
) -> list[tuple[str, dict[str, Any]]]:
    """合并分批结果。同一对段落保留置信度更高的一条边。"""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for group in groups:
        for source_id, edge in group:
            key = (source_id, str(edge.get("node_id") or ""))
            current = best.get(key)
            if current is None:
                best[key] = edge
                order.append(key)
            elif float(edge.get("confidence") or 0) > float(current.get("confidence") or 0):
                best[key] = edge
    return [(key[0], best[key]) for key in order]


def assemble_note_dependencies(
    edges: list[tuple[str, dict[str, Any]]],
    node_ids: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """丢弃越界边和自环，并为没有边的段落补空数组。"""
    node_id_set = set(node_ids)
    dependencies: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    seen: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    discarded = 0
    for source_id, edge in edges:
        target = str(edge.get("node_id") or "")
        if source_id not in node_id_set or target not in node_id_set or target == source_id or target in seen[source_id]:
            discarded += 1
            continue
        dependencies[source_id].append(edge)
        seen[source_id].add(target)
    return dependencies, discarded


def build_note_document_card(
    edges: list[tuple[str, dict[str, Any]]],
    node_ids: list[str],
    doc_content_hash: str,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    dependencies, discarded = assemble_note_dependencies(edges, node_ids)
    if discarded:
        logging.getLogger("rag.dependency_graph").warning("依赖边校验丢弃 | 数量=%s", discarded)
    return {
        "status": "success",
        "doc_content_hash": doc_content_hash,
        "expected_node_count": len(node_ids),
        "processed_node_count": len(node_ids),
        "model": model,
        "prompt_version": prompt_version,
        "generated_at": utc_now(),
        "error": None,
        "discarded_edge_count": discarded,
        "node_dependencies": dependencies,
    }


def build_aggregation_input(nodes: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: int((getattr(item, "metadata", {}) or {}).get("chunk_index", 0))):
        metadata = dict(getattr(node, "metadata", {}) or {})
        result.append(
            {
                "node_id": node_key(node),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "archive_path": str(metadata.get("archive_path") or ""),
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
                    f"归档：{source_metadata.get('archive_path') or ''}",
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
    prompt_path = Path(path) if path is not None else Path("prompts/dependency_graph_v5.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_DEPENDENCY_PROMPT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


DEFAULT_DEPENDENCY_PROMPT = """
你是文档内部依赖整理器。输入只包含同一篇文档的标注笔记，不包含各段全文。
subject 描述这段是什么，stated_rules 是这段替别人定下的规矩。
chunks 是这篇文档的全部笔记，整理其中每一条规矩。
被约束的段落指向规矩所在段落。用种类、标题、路径、配置键和适用范围判断是否覆盖。
无法确认时不要连边。不允许自己依赖自己或使用不存在的 node_id。
relation_type 只能取 prerequisite、constraint、failure_handling、follow_up。
每条边必须包含 node_id、relation_type、required、0 到 1 的 confidence 和非空 reason。
reason 必须引用规矩的 evidence。只输出有边的 source。
只返回严格 JSON：{"node_dependencies": {"source-node-id": []}}。
""".strip()
