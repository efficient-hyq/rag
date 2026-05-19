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

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_chunks(self, nodes: Iterable[Any]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as file:
            for node in nodes:
                record = {
                    "node_id": node_key(node),
                    "text_hash": text_hash(str(getattr(node, "text", ""))),
                    "metadata": dict(getattr(node, "metadata", {}) or {}),
                    "text": str(getattr(node, "text", "")),
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

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


def print_progress(snapshot: ProgressSnapshot) -> None:
    print(
        f"{snapshot.stage}进度 {snapshot.done}/{snapshot.total} | "
        f"失败 {snapshot.failed} | {snapshot.percent:.1f}%",
        file=sys.stdout,
        flush=True,
    )
