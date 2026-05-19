from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from rag.retrieval.ranking import DualStageReranker, QueryRewriter, RuleBasedReranker
from rag.retrieval.tokenization import tokenize_technical_text


@dataclass
class RetrievalResult:
    """一次查询的完整输出，兼顾调试信息和最终展示结果。"""

    query: str
    rewritten_queries: list[str]
    candidates: list["RetrievalCandidate"]
    top_candidates: list["RetrievalCandidate"]


@dataclass
class RetrievedRouteHit:
    """单一路由命中的原始记录，用于后续聚合成候选 chunk。"""

    node_id: str
    route: str
    rank: int
    raw_score: float
    query_text: str
    is_rewritten: bool


@dataclass
class RetrievalCandidate:
    """按 node_id 聚合后的候选 chunk，是重排阶段的基本单元。"""

    node_id: str
    text: str
    metadata: dict[str, Any]
    hits: list[RetrievedRouteHit] = field(default_factory=list)
    fused_score: float = 0.0
    llm_score: float | None = None
    final_score: float = 0.0
    is_neighbor: bool = False
    center_node_id: str | None = None


class MetadataIndex:
    """查询期使用的轻量元数据索引，主要支持 chunk 定位和相邻扩展。"""

    def __init__(self, metadata_by_id: dict[str, dict[str, Any]]) -> None:
        self.metadata_by_id = {node_id: dict(metadata) for node_id, metadata in metadata_by_id.items()}
        self.doc_chunk_to_node_id: dict[tuple[str, int], str] = {}
        for node_id, metadata in self.metadata_by_id.items():
            doc_id = str(metadata.get("doc_id") or "")
            chunk_index = metadata.get("chunk_index")
            if not doc_id or chunk_index is None:
                continue
            self.doc_chunk_to_node_id[(doc_id, int(chunk_index))] = node_id

    @classmethod
    def from_file(cls, path: str | Path) -> "MetadataIndex":
        metadata_path = Path(path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata 文件不存在: {metadata_path}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("metadata.json 格式非法，顶层必须是对象")
        normalized = {str(node_id): dict(item) for node_id, item in raw.items() if isinstance(item, dict)}
        return cls(normalized)

    def get_neighbor_node_ids(self, node_id: str, radius: int = 1) -> list[str]:
        metadata = self.metadata_by_id.get(node_id, {})
        doc_id = str(metadata.get("doc_id") or "")
        chunk_index = metadata.get("chunk_index")
        if not doc_id or chunk_index is None:
            return []
        chunk_index = int(chunk_index)

        neighbor_ids: list[str] = []
        for offset in range(1, max(radius, 0) + 1):
            previous_id = self.doc_chunk_to_node_id.get((doc_id, chunk_index - offset))
            next_id = self.doc_chunk_to_node_id.get((doc_id, chunk_index + offset))
            if previous_id is not None:
                neighbor_ids.append(previous_id)
            if next_id is not None:
                neighbor_ids.append(next_id)
        return neighbor_ids


def aggregate_route_hits(
    hits: list[RetrievedRouteHit],
    metadata_by_id: dict[str, dict[str, Any]],
) -> list[RetrievalCandidate]:
    """将多路命中按 node_id 聚合，形成后续可重排的候选集合。"""
    aggregated: dict[str, RetrievalCandidate] = {}
    for hit in hits:
        metadata = dict(metadata_by_id.get(hit.node_id, {}))
        candidate = aggregated.get(hit.node_id)
        if candidate is None:
            candidate = RetrievalCandidate(
                node_id=hit.node_id,
                text=str(metadata.get("text") or ""),
                metadata=metadata,
            )
            aggregated[hit.node_id] = candidate
        candidate.hits.append(hit)
    return list(aggregated.values())


def expand_neighbor_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    metadata_index: MetadataIndex,
    radius: int = 1,
    center_top_k: int = 5,
) -> list[RetrievalCandidate]:
    """围绕粗排前若干中心 chunk 扩展相邻上下文，补足回答时所需语境。"""
    expanded = list(candidates)
    seen_ids = {candidate.node_id for candidate in candidates}
    centers = candidates[:center_top_k]

    for center_rank, center in enumerate(centers, start=1):
        for neighbor_id in metadata_index.get_neighbor_node_ids(center.node_id, radius=radius):
            if neighbor_id in seen_ids:
                continue
            metadata = dict(metadata_index.metadata_by_id.get(neighbor_id, {}))
            neighbor = RetrievalCandidate(
                node_id=neighbor_id,
                text=str(metadata.get("text") or ""),
                metadata=metadata,
                hits=[
                    RetrievedRouteHit(
                        node_id=neighbor_id,
                        route="neighbor_expand",
                        rank=center_rank,
                        raw_score=0.0,
                        query_text=query,
                        is_rewritten=False,
                    )
                ],
                is_neighbor=True,
                center_node_id=center.node_id,
            )
            expanded.append(neighbor)
            seen_ids.add(neighbor_id)
    return expanded


class BM25KeywordRetriever:
    """基于本地 BM25 索引的关键词召回器。"""

    def __init__(self, bm25_path: str | Path, tokenizer=tokenize_technical_text) -> None:
        self.bm25_path = Path(bm25_path)
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"BM25 索引不存在: {self.bm25_path}")
        self.tokenizer = tokenizer
        with self.bm25_path.open("rb") as file:
            self.payload = pickle.load(file)

    def retrieve(self, query: str, top_k: int = 5, is_rewritten: bool = False) -> list[RetrievedRouteHit]:
        query_tokens = self.tokenizer(query)
        scores = self.payload.bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.payload.node_ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        hits: list[RetrievedRouteHit] = []
        for rank, (node_id, score) in enumerate(ranked[:top_k], start=1):
            hits.append(
                RetrievedRouteHit(
                    node_id=str(node_id),
                    route="bm25",
                    rank=rank,
                    raw_score=float(score),
                    query_text=query,
                    is_rewritten=is_rewritten,
                )
            )
        return hits


class HybridRetriever:
    """多路混合召回器，负责把向量、BM25、改写与重排串成完整链路。"""

    def __init__(
        self,
        embedder: Any,
        storage_dir: str | Path = "storage",
        chroma_client: Any | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: DualStageReranker | None = None,
        content_top_k: int = 8,
        summary_top_k: int = 6,
        bm25_top_k: int = 8,
        rewrite_content_top_k: int = 4,
        rewrite_summary_top_k: int = 3,
        rewrite_bm25_top_k: int = 4,
        neighbor_enabled: bool = True,
        neighbor_radius: int = 1,
        center_top_k: int = 5,
        final_top_n: int = 5,
    ) -> None:
        self.embedder = embedder
        self.storage_dir = Path(storage_dir)
        self.chroma_dir = self.storage_dir / "chroma"
        self.content_top_k = content_top_k
        self.summary_top_k = summary_top_k
        self.bm25_top_k = bm25_top_k
        self.rewrite_content_top_k = rewrite_content_top_k
        self.rewrite_summary_top_k = rewrite_summary_top_k
        self.rewrite_bm25_top_k = rewrite_bm25_top_k
        self.neighbor_enabled = neighbor_enabled
        self.neighbor_radius = neighbor_radius
        self.center_top_k = center_top_k
        self.final_top_n = final_top_n
        self.query_rewriter = query_rewriter
        self.reranker = reranker or DualStageReranker(rule_reranker=RuleBasedReranker(), llm_enabled=False)
        self.rule_reranker = self.reranker.rule_reranker

        self.chroma_client = chroma_client or chromadb.PersistentClient(path=str(self.chroma_dir))
        self.content_collection = self.chroma_client.get_collection(name="content_vec")
        self.summary_collection = self.chroma_client.get_collection(name="summary_vec")
        self.metadata_index = MetadataIndex.from_file(self.storage_dir / "metadata.json")
        self.bm25_retriever = BM25KeywordRetriever(self.storage_dir / "bm25.pkl")

    def retrieve(self, query: str) -> RetrievalResult:
        """执行在线查询主流程：原始召回 -> 改写补召回 -> 粗排 -> 相邻扩展 -> 最终重排。"""
        logger = logging.getLogger("rag.retriever")
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        logger.info("召回开始 | query=%s", query)
        query_embedding = self.embedder.get_text_embedding(query)
        hits = self._retrieve_for_query(query, query_embedding, is_rewritten=False)
        logger.info("原始召回完成 | 命中数=%s", len(hits))

        rewritten_queries = self.query_rewriter.rewrite(query) if self.query_rewriter is not None else []
        logger.info("改写结果 | 数量=%s", len(rewritten_queries))
        for rewritten_query in rewritten_queries:
            rewritten_embedding = self.embedder.get_text_embedding(rewritten_query)
            rewritten_hits = self._retrieve_for_query(
                rewritten_query,
                rewritten_embedding,
                is_rewritten=True,
                content_top_k=self.rewrite_content_top_k,
                summary_top_k=self.rewrite_summary_top_k,
                bm25_top_k=self.rewrite_bm25_top_k,
            )
            hits.extend(rewritten_hits)
            logger.info("改写召回完成 | query=%s | 命中数=%s", rewritten_query, len(rewritten_hits))

        candidates = aggregate_route_hits(hits, self.metadata_index.metadata_by_id)
        logger.info("候选聚合完成 | 候选数=%s", len(candidates))
        coarse_ranked = self.rule_reranker.rerank(candidates)
        if self.neighbor_enabled:
            candidates = expand_neighbor_candidates(
                query,
                coarse_ranked,
                self.metadata_index,
                radius=self.neighbor_radius,
                center_top_k=self.center_top_k,
            )
            logger.info("邻居扩展完成 | 候选数=%s", len(candidates))

        final_ranked = self.reranker.rerank(query, candidates)
        top_candidates = final_ranked[: self.final_top_n]
        logger.info("重排完成 | 最终候选数=%s | top_n=%s", len(final_ranked), len(top_candidates))
        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten_queries,
            candidates=final_ranked,
            top_candidates=top_candidates,
        )

    def _retrieve_for_query(
        self,
        query: str,
        query_embedding: list[float],
        is_rewritten: bool,
        content_top_k: int | None = None,
        summary_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> list[RetrievedRouteHit]:
        logger = logging.getLogger("rag.retriever")
        hits: list[RetrievedRouteHit] = []
        hits.extend(
            self._query_vector_route(
                collection=self.content_collection,
                route="content_vec",
                query=query,
                query_embedding=query_embedding,
                top_k=content_top_k or self.content_top_k,
                is_rewritten=is_rewritten,
            )
        )
        hits.extend(
            self._query_vector_route(
                collection=self.summary_collection,
                route="summary_vec",
                query=query,
                query_embedding=query_embedding,
                top_k=summary_top_k or self.summary_top_k,
                is_rewritten=is_rewritten,
            )
        )
        hits.extend(
            self.bm25_retriever.retrieve(
                query=query,
                top_k=bm25_top_k or self.bm25_top_k,
                is_rewritten=is_rewritten,
            )
        )
        logger.info(
            "单查询召回完成 | query=%s | content=%s | summary=%s | bm25=%s | 总命中=%s",
            query,
            content_top_k or self.content_top_k,
            summary_top_k or self.summary_top_k,
            bm25_top_k or self.bm25_top_k,
            len(hits),
        )
        return hits

    @staticmethod
    def _query_vector_route(
        collection: Any,
        route: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        is_rewritten: bool,
    ) -> list[RetrievedRouteHit]:
        if not query_embedding:
            return []
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances"],
        )
        ids = result.get("ids", [[]])
        distances = result.get("distances", [[]])
        route_ids = ids[0] if ids else []
        route_distances = distances[0] if distances else []

        hits: list[RetrievedRouteHit] = []
        for rank, node_id in enumerate(route_ids, start=1):
            distance = route_distances[rank - 1] if rank - 1 < len(route_distances) else 0.0
            hits.append(
                RetrievedRouteHit(
                    node_id=str(node_id),
                    route=route,
                    rank=rank,
                    raw_score=float(-distance),
                    query_text=query,
                    is_rewritten=is_rewritten,
                )
            )
        return hits
