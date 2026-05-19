from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from rag.shared.checkpoints import CheckpointStore, ProgressSnapshot, node_key, print_progress, text_hash


MAX_OPENAI_COMPATIBLE_EMBEDDING_BATCH_SIZE = 10


class EmbeddingModel(Protocol):
    cache_identity: str

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def get_text_embedding(self, text: str) -> list[float]:
        ...


@dataclass
class OpenAICompatibleEmbedding:
    """OpenAI 兼容 Embedding 适配器。"""

    api_key: str
    base_url: str
    model: str
    max_batch_size: int = MAX_OPENAI_COMPATIBLE_EMBEDDING_BATCH_SIZE
    client: OpenAI | None = None

    @property
    def cache_identity(self) -> str:
        return f"openai-compatible:{self.model}"

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("向量化批大小必须大于 0")
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for offset in range(0, len(texts), self.max_batch_size):
            batch = texts[offset : offset + self.max_batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(list(item.embedding) for item in response.data)
        return embeddings

    def get_text_embedding(self, text: str) -> list[float]:
        if not text:
            return []
        response = self.client.embeddings.create(model=self.model, input=[text])
        return list(response.data[0].embedding)


def build_openai_compatible_embedder(
    api_key: str,
    base_url: str,
    model: str,
) -> EmbeddingModel:
    """创建 OpenAI 兼容 Embedding 模型。"""
    return OpenAICompatibleEmbedding(api_key=api_key, base_url=base_url, model=model)


def embed_texts(embedder: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    """批量向量化文本，统一空输入处理。"""
    if not texts:
        return []
    return embedder.get_text_embedding_batch(texts)


def embed_nodes_with_checkpoint(
    embedder: EmbeddingModel,
    nodes: list[object],
    texts: list[str],
    checkpoint: CheckpointStore,
    route: str,
    batch_size: int = MAX_OPENAI_COMPATIBLE_EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """按批次向量化并写入检查点，重跑时跳过已完成 chunk。"""
    logger = logging.getLogger("rag.embedder")
    if len(nodes) != len(texts):
        raise ValueError("nodes 与 texts 数量必须一致")

    identity = getattr(embedder, "cache_identity", embedder.__class__.__name__)
    cached = checkpoint.load_embeddings(route)
    embeddings_by_key = dict(cached)
    pending = [
        (node, text)
        for node, text in zip(nodes, texts)
        if _embedding_cache_key(node, route, identity) not in embeddings_by_key
    ]
    failed = 0
    completed = len(nodes) - len(pending)

    if show_progress:
        print_progress(ProgressSnapshot(f"{route}向量化", len(nodes), completed, failed))
    logger.info("%s向量化开始 | 总数=%s | 已完成=%s", route, len(nodes), completed)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        batch_nodes = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]
        try:
            batch_embeddings = embed_texts(embedder, batch_texts)
        except Exception as exc:
            failed += len(batch)
            logger.warning("%s向量化失败 | 批次起点=%s | 批大小=%s | error=%s", route, offset, len(batch), exc)
            for node in batch_nodes:
                checkpoint.append_embedding(
                    route,
                    _embedding_cache_key(node, route, identity),
                    [],
                    status="failed",
                    error=str(exc),
                )
            if show_progress:
                print_progress(ProgressSnapshot(f"{route}向量化", len(nodes), completed, failed))
            raise

        if len(batch_embeddings) != len(batch_nodes):
            raise ValueError(f"{route} embedding 返回数量与批次数量不一致")

        for node, embedding in zip(batch_nodes, batch_embeddings):
            key = _embedding_cache_key(node, route, identity)
            embeddings_by_key[key] = embedding
            checkpoint.append_embedding(route, key, embedding)
            completed += 1
        logger.info("%s向量化进度 | 已完成=%s/%s", route, completed, len(nodes))
        if show_progress:
            print_progress(ProgressSnapshot(f"{route}向量化", len(nodes), completed, failed))

    missing = [
        node_key(node)
        for node in nodes
        if _embedding_cache_key(node, route, identity) not in embeddings_by_key
    ]
    if missing:
        raise RuntimeError(f"{route} 向量缺失，无法入库: {missing[:5]}")
    logger.info("%s向量化结束 | 总数=%s | 失败=%s", route, len(nodes), failed)
    return [embeddings_by_key[_embedding_cache_key(node, route, identity)] for node in nodes]


def _embedding_cache_key(node: object, route: str, identity: str) -> str:
    key = f"{node_key(node)}|embedding|{route}|{identity}"
    if route == "summary":
        metadata = dict(getattr(node, "metadata", {}) or {})
        summary = str(metadata.get("summary") or "")
        return f"{key}|summary:{text_hash(summary)}"
    return key
