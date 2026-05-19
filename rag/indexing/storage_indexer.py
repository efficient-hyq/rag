from __future__ import annotations

import json
import logging
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag.retrieval.tokenization import tokenize_technical_text


@dataclass
class StoredBM25Index:
    """可序列化的 BM25 索引载荷。"""

    node_ids: list[str]
    tokenized_corpus: list[list[str]]
    bm25: Any


@dataclass
class SimpleBM25Okapi:
    """rank-bm25 未安装时的最小可用 BM25 实现。"""

    tokenized_corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_lengths = [len(document) for document in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.idf = self._build_idf()

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for document, doc_len in zip(self.tokenized_corpus, self.doc_lengths):
            frequencies = Counter(document)
            score = 0.0
            for token in query_tokens:
                freq = frequencies.get(token, 0)
                if freq == 0:
                    continue
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(token, 0.0) * freq * (self.k1 + 1) / denominator
            scores.append(score)
        return scores

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.tokenized_corpus)
        document_frequency: Counter[str] = Counter()
        for document in self.tokenized_corpus:
            document_frequency.update(set(document))
        return {
            token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }


@dataclass(frozen=True)
class IndexResult:
    node_count: int
    content_collection: str
    summary_collection: str
    bm25_path: Path
    metadata_path: Path


class MultiRouteIndexer:
    """将 chunk 同步写入向量库、BM25 与 metadata.json。"""

    def __init__(
        self,
        storage_dir: str | Path = "storage",
        chroma_client: Any | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.chroma_dir = self.storage_dir / "chroma"
        self.bm25_path = self.storage_dir / "bm25.pkl"
        self.metadata_path = self.storage_dir / "metadata.json"
        self.chroma_client = chroma_client
        self.tokenizer = tokenizer or tokenize_technical_text

    def index(
        self,
        nodes: list[Any],
        content_embeddings: list[list[float]],
        summary_embeddings: list[list[float]],
    ) -> IndexResult:
        logger = logging.getLogger("rag.indexer")
        self._validate(nodes, content_embeddings, summary_embeddings)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info("入库开始 | chunk数=%s | storage_dir=%s", len(nodes), self.storage_dir)

        normalized_nodes = [normalize_node(node) for node in nodes]
        content_collection = self._collection("content_vec")
        summary_collection = self._collection("summary_vec")

        ids = [node["node_id"] for node in normalized_nodes]
        content_documents = [node["text"] for node in normalized_nodes]
        summary_documents = [node["summary"] for node in normalized_nodes]
        chroma_metadatas = [sanitize_chroma_metadata(node["metadata"]) for node in normalized_nodes]

        content_collection.upsert(
            ids=ids,
            embeddings=content_embeddings,
            documents=content_documents,
            metadatas=chroma_metadatas,
        )
        summary_collection.upsert(
            ids=ids,
            embeddings=summary_embeddings,
            documents=summary_documents,
            metadatas=chroma_metadatas,
        )
        logger.info("向量库写入完成 | collection=content_vec,summary_vec | chunk数=%s", len(nodes))

        self._write_bm25(normalized_nodes)
        self._write_metadata(normalized_nodes)
        logger.info("索引文件写入完成 | bm25=%s | metadata=%s", self.bm25_path, self.metadata_path)
        return IndexResult(
            node_count=len(nodes),
            content_collection="content_vec",
            summary_collection="summary_vec",
            bm25_path=self.bm25_path,
            metadata_path=self.metadata_path,
        )

    def _collection(self, name: str) -> Any:
        client = self.chroma_client or self._build_chroma_client()
        return client.get_or_create_collection(name=name)

    def _build_chroma_client(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("缺少 chromadb，请先安装 requirements.txt 中的依赖") from exc
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self.chroma_client

    def _write_bm25(self, nodes: list[dict[str, Any]]) -> None:
        tokenized_corpus = [tokens_for_bm25(node, self.tokenizer) for node in nodes]
        bm25 = build_bm25(tokenized_corpus)
        payload = StoredBM25Index(
            node_ids=[node["node_id"] for node in nodes],
            tokenized_corpus=tokenized_corpus,
            bm25=bm25,
        )
        with self.bm25_path.open("wb") as file:
            pickle.dump(payload, file)

    def _write_metadata(self, nodes: list[dict[str, Any]]) -> None:
        metadata = {}
        for node in nodes:
            item = {"text": node["text"]}
            item.update(node["metadata"])
            metadata[node["node_id"]] = item
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _validate(
        nodes: list[Any],
        content_embeddings: list[list[float]],
        summary_embeddings: list[list[float]],
    ) -> None:
        if len(content_embeddings) != len(nodes):
            raise ValueError("content_embeddings 数量必须与 nodes 数量一致")
        if len(summary_embeddings) != len(nodes):
            raise ValueError("summary_embeddings 数量必须与 nodes 数量一致")


def normalize_node(node: Any) -> dict[str, Any]:
    node_id = str(getattr(node, "node_id", None) or getattr(node, "id_", None) or "")
    if not node_id:
        raise ValueError("node 缺少 node_id/id_")
    text = str(getattr(node, "text", None) or node.get_content())
    metadata = dict(getattr(node, "metadata", {}) or {})
    summary = str(metadata.get("summary") or "")
    return {
        "node_id": node_id,
        "text": text,
        "summary": summary,
        "metadata": metadata,
    }


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = str(value)
    return sanitized


def tokens_for_bm25(node: dict[str, Any], tokenizer: Callable[[str], list[str]]) -> list[str]:
    metadata = node["metadata"]
    text_parts = [node["text"], node["summary"]]
    text_parts.extend(str(item) for item in metadata.get("keywords", []) if item)
    text_parts.extend(str(item) for item in metadata.get("tags", []) if item)
    return tokenizer(" ".join(text_parts))


def build_bm25(tokenized_corpus: list[list[str]]) -> Any:
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return SimpleBM25Okapi(tokenized_corpus)
