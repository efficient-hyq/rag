from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProgressSnapshot:
    """单个阶段的进度快照。"""

    stage: str
    total: int
    done: int
    failed: int = 0

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 100.0
        return self.done / self.total * 100


class CheckpointStore:
    """离线索引检查点存储，保证标注和向量化可断点续跑。"""

    def __init__(self, storage_dir: str | Path) -> None:
        self.root = Path(storage_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.root / "chunks.jsonl"
        self.annotations_path = self.root / "annotations.jsonl"
        self.content_embeddings_path = self.root / "embeddings_content.jsonl"
        self.summary_embeddings_path = self.root / "embeddings_summary.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.document_index_state_path = self.root / "document_index_state.json"

    def load_document_index_state(self) -> dict[str, Any]:
        if not self.document_index_state_path.exists():
            return {"docs": {}}
        return json.loads(self.document_index_state_path.read_text(encoding="utf-8"))

    def save_document_index_state(self, payload: dict[str, Any]) -> None:
        self.document_index_state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_chunks(self, nodes: Iterable[Any]) -> None:
        records = []
        for node in nodes:
            records.append(
                {
                    "node_id": node_key(node),
                    "text_hash": text_hash(str(getattr(node, "text", ""))),
                    "metadata": dict(getattr(node, "metadata", {}) or {}),
                    "text": str(getattr(node, "text", "")),
                }
            )
        self.upsert_chunk_records(records)

    def load_chunk_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.chunks_path.exists():
            return records
        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                node_id = str(record.get("node_id") or "")
                if node_id:
                    records[node_id] = record
        return records

    def upsert_chunk_records(self, records: Iterable[dict[str, Any]]) -> None:
        current = self.load_chunk_records()
        for record in records:
            node_id = str(record.get("node_id") or "")
            if node_id:
                current[node_id] = dict(record)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8") as file:
            for record in current.values():
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def remove_node_records(self, node_ids: set[str]) -> None:
        if not node_ids:
            return
        self._rewrite_chunk_records_without(node_ids)
        self._rewrite_jsonl_without_node_keys(self.annotations_path, node_ids)
        self._rewrite_jsonl_without_node_keys(self.content_embeddings_path, node_ids)
        self._rewrite_jsonl_without_node_keys(self.summary_embeddings_path, node_ids)

    def load_raw_records(self, path: Path) -> dict[str, dict[str, Any]]:
        return self._load_jsonl_by_key(path)

    def load_annotations(self) -> dict[str, dict[str, Any]]:
        return {
            key: dict(record.get("annotation") or {})
            for key, record in self._load_jsonl_by_key(self.annotations_path).items()
            if record.get("status") == "success"
        }

    def append_annotation(
        self,
        key: str,
        annotation: dict[str, Any],
        status: str = "success",
        error: str | None = None,
    ) -> None:
        self._upsert_jsonl(
            self.annotations_path,
            {
                "key": key,
                "status": status,
                "annotation": annotation,
                "error": error,
            },
        )

    def load_embeddings(self, route: str) -> dict[str, list[float]]:
        path = self._embedding_path(route)
        return {
            key: list(record.get("embedding") or [])
            for key, record in self._load_jsonl_by_key(path).items()
            if record.get("status") == "success"
        }

    def append_embedding(
        self,
        route: str,
        key: str,
        embedding: list[float],
        status: str = "success",
        error: str | None = None,
    ) -> None:
        self._upsert_jsonl(
            self._embedding_path(route),
            {
                "key": key,
                "status": status,
                "embedding": embedding,
                "error": error,
            },
        )

    def _embedding_path(self, route: str) -> Path:
        if route == "content":
            return self.content_embeddings_path
        if route == "summary":
            return self.summary_embeddings_path
        raise ValueError(f"未知向量路由: {route}")

    def _upsert_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load_jsonl_by_key(path)
        key = str(record.get("key") or "")
        if not key:
            raise ValueError("checkpoint 记录缺少 key")
        records[key] = record
        with path.open("w", encoding="utf-8") as file:
            for item in records.values():
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _rewrite_chunk_records_without(self, node_ids: set[str]) -> None:
        records = {
            node_id: record
            for node_id, record in self.load_chunk_records().items()
            if node_id not in node_ids
        }
        with self.chunks_path.open("w", encoding="utf-8") as file:
            for record in records.values():
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rewrite_jsonl_without_node_keys(self, path: Path, node_ids: set[str]) -> None:
        records = {
            key: record
            for key, record in self._load_jsonl_by_key(path).items()
            if not _checkpoint_key_matches_node(key, node_ids)
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records.values():
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_jsonl_by_key(path: Path) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record.get("key") or "")
                if key:
                    records[key] = record
        return records


def node_key(node: Any) -> str:
    """返回 chunk 的稳定键。"""
    return str(getattr(node, "node_id", None) or getattr(node, "id_", None) or "")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stable_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    raw = f"{doc_id}|{chunk_index}|{text_hash(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _checkpoint_key_matches_node(key: str, node_ids: set[str]) -> bool:
    node_id = key.split("|", 1)[0]
    return node_id in node_ids


def print_progress(snapshot: ProgressSnapshot) -> None:
    print(
        f"{snapshot.stage}进度 {snapshot.done}/{snapshot.total} | "
        f"失败 {snapshot.failed} | {snapshot.percent:.1f}%",
        file=sys.stdout,
        flush=True,
    )
