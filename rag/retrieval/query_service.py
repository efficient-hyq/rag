from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from rag.config import QueryConfig
from rag.indexing.embedding_client import build_openai_compatible_embedder
from rag.retrieval.answer_generator import AnswerGenerator, AnswerReference
from rag.retrieval.ranking import DualStageReranker, LLMReranker, QueryRewriter, RuleBasedReranker
from rag.retrieval.retriever import HybridRetriever, RetrievalCandidate
from rag.shared.logging_utils import log_phase


@dataclass(frozen=True)
class QueryResponse:
    """完整查询结果，包含检索与答案生成。"""

    question: str
    rewritten_queries: list[str]
    answer: str
    references: list[AnswerReference]
    candidates: list[RetrievalCandidate]
    top_candidates: list[RetrievalCandidate]


def build_query_service(config: QueryConfig | None = None) -> "QueryService":
    cfg = config or QueryConfig.from_env()
    embedder = build_openai_compatible_embedder(
        api_key=cfg.embedding.api_key or "",
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
    )
    llm_client = _build_llm_client(cfg)
    retriever = HybridRetriever(
        embedder=embedder,
        storage_dir=cfg.paths.storage_dir,
        query_rewriter=build_query_rewriter(cfg, llm_client),
        reranker=build_dual_stage_reranker(cfg, llm_client),
        content_top_k=cfg.routes.content_top_k,
        summary_top_k=cfg.routes.summary_top_k,
        bm25_top_k=cfg.routes.bm25_top_k,
        rewrite_content_top_k=cfg.routes.rewrite_content_top_k,
        rewrite_summary_top_k=cfg.routes.rewrite_summary_top_k,
        rewrite_bm25_top_k=cfg.routes.rewrite_bm25_top_k,
        neighbor_enabled=cfg.routes.neighbor_expand_enabled,
        neighbor_radius=cfg.routes.neighbor_radius,
        center_top_k=cfg.routes.center_top_k,
        final_top_n=cfg.ranking.final_top_n,
    )
    answer_generator = AnswerGenerator(
        client=llm_client,
        model=cfg.answer.model,
        enabled=cfg.answer.enabled,
        context_top_k=cfg.answer.context_top_k,
        max_context_chars=cfg.answer.max_context_chars,
    )
    return QueryService(retriever=retriever, answer_generator=answer_generator)


class QueryService:
    """在线查询服务，封装召回、重排和答案生成。"""

    def __init__(self, retriever: HybridRetriever, answer_generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

    def query(self, question: str | None) -> QueryResponse:
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        logger = logging.getLogger("rag.query")
        with log_phase(logger, "在线查询", question=question):
            retrieval_result = self.retriever.retrieve(question)
            generated = self.answer_generator.generate(question, retrieval_result.top_candidates)
        logger.info(
            "查询完成 | 改写数=%s | 候选数=%s | TopN=%s | 引用数=%s",
            len(retrieval_result.rewritten_queries),
            len(retrieval_result.candidates),
            len(retrieval_result.top_candidates),
            len(generated.references),
        )
        return QueryResponse(
            question=question,
            rewritten_queries=retrieval_result.rewritten_queries,
            answer=generated.answer,
            references=generated.references,
            candidates=retrieval_result.candidates,
            top_candidates=retrieval_result.top_candidates,
        )


def render_query_response(response: QueryResponse) -> str:
    """将完整查询结果格式化为便于命令行查看的文本。"""
    lines = [f"question={response.question}"]
    if response.rewritten_queries:
        lines.append(f"rewrites={','.join(response.rewritten_queries)}")
    else:
        lines.append("rewrites=<none>")

    lines.append("")
    lines.append("answer:")
    lines.append(response.answer)
    lines.append("")
    lines.append("references:")
    if response.references:
        for index, item in enumerate(response.references, start=1):
            lines.append(
                f"{index}. node_id={item.node_id} file={item.file_name} chunk={item.chunk_index} score={item.score:.4f}"
            )
    else:
        lines.append("<none>")

    lines.append("")
    lines.append("top_candidates:")
    for index, candidate in enumerate(response.top_candidates, start=1):
        routes = ",".join(dict.fromkeys(hit.route for hit in candidate.hits))
        file_name = str(candidate.metadata.get("file_name") or candidate.metadata.get("doc_id") or "unknown")
        chunk_index = candidate.metadata.get("chunk_index", "?")
        lines.append(
            f"{index}. node_id={candidate.node_id} route={routes} score={candidate.final_score:.4f} file={file_name} chunk={chunk_index}"
        )
    return "\n".join(lines)


def build_query_rewriter(cfg: QueryConfig, llm_client: OpenAI | None) -> QueryRewriter | None:
    if not cfg.rewrite.enabled or llm_client is None:
        return None
    return QueryRewriter(
        client=llm_client,
        model=cfg.rewrite.model,
        enabled=True,
        rewrite_limit=cfg.rewrite.limit,
    )


def build_dual_stage_reranker(cfg: QueryConfig, llm_client: OpenAI | None) -> DualStageReranker:
    rule_reranker = RuleBasedReranker()
    llm_reranker = None
    if cfg.ranking.llm_enabled and llm_client is not None:
        llm_reranker = LLMReranker(client=llm_client, model=cfg.ranking.llm_model)
    return DualStageReranker(
        rule_reranker=rule_reranker,
        llm_reranker=llm_reranker,
        llm_enabled=cfg.ranking.llm_enabled,
        llm_top_n=cfg.ranking.llm_top_n,
    )


def _build_llm_client(cfg: QueryConfig) -> OpenAI | None:
    if not cfg.llm.api_key:
        return None
    return OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)
