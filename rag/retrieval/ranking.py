from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from rag.retrieval.retriever import RetrievalCandidate


class RuleBasedReranker:
    """规则粗排器，基于 RRF、多路命中奖励和相邻降权生成初始顺序。"""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def score(self, candidate: "RetrievalCandidate") -> float:
        """计算单个候选的规则分数，并把结果回写到候选对象。"""
        score = 0.0
        routes = {hit.route for hit in candidate.hits}
        rewritten_hits = sum(1 for hit in candidate.hits if hit.is_rewritten)

        for hit in candidate.hits:
            score += 1.0 / (self.rrf_k + max(hit.rank, 1))

        if len(routes) > 1:
            score += 0.05 * (len(routes) - 1)
        if rewritten_hits > 1:
            score += 0.02 * (rewritten_hits - 1)

        coherence = str(candidate.metadata.get("coherence") or "").lower()
        if coherence == "high":
            score += 0.03
        elif coherence == "low":
            score -= 0.03

        if candidate.is_neighbor:
            score -= 0.06

        candidate.fused_score = score
        return score

    def rerank(self, candidates: list["RetrievalCandidate"]) -> list["RetrievalCandidate"]:
        """批量计算规则分并按降序返回，作为第一阶段粗排结果。"""
        for candidate in candidates:
            candidate.final_score = self.score(candidate)
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)


class QueryRewriter:
    """查询改写器，把用户问题扩展成若干更利于召回的子查询。"""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "qwen3.6-plus",
        enabled: bool = True,
        rewrite_limit: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.rewrite_limit = rewrite_limit

    def rewrite(self, query: str) -> list[str]:
        """调用 LLM 生成改写查询，并解析为查询列表。"""
        if not self.enabled or self.client is None or not query:
            return []
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 检索改写器，负责把用户问题改写成更适合文档召回的检索子查询。"
                        "你的目标是缩小“用户表达”和“文档表述”之间的词汇差异，提高向量召回和关键词召回命中率。"
                        "不要回答问题，不要解释，不要补充结论，不要输出 Markdown。"
                        "请严格遵守以下规则："
                        "1. 只输出 2 到 4 个检索子查询，优先覆盖主题词、业务动作、关键对象、接口或配置项；"
                        "2. 保留原问题中的产品名、接口名、配置项、错误码、类名、方法名、英文技术术语和版本号，不要随意改写；"
                        "3. 如果原问题过于口语化，请补成更利于检索的关键词表达；"
                        "4. 如果原问题已经很明确，可以只做轻量改写，不要过度发散；"
                        "5. 子查询之间要有差异，但必须与原问题强相关，禁止引入无关上下文；"
                        "6. 输出应偏向文档检索表达，例如流程、参数、回调、配置、校验、异常处理、接口调用；"
                        "7. 每个子查询保持简洁，通常不超过 18 个字或一个短句；"
                        f"8. 只返回 JSON，格式必须是 {{\"queries\": [\"...\"]}}，最多 {self.rewrite_limit} 条。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"请基于下面的原始问题生成检索子查询。\n原始问题：{query}",
                },
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return _parse_query_rewrite(content, self.rewrite_limit)


class LLMReranker:
    """第二阶段语义精排器，只对粗排前若干候选做更细粒度判断。"""

    def __init__(self, client: OpenAI | None = None, model: str = "qwen3.6-plus") -> None:
        self.client = client
        self.model = model

    def rerank(self, query: str, candidates: list["RetrievalCandidate"]) -> dict[str, float]:
        """返回 node_id 到语义分数的映射，供第二阶段融合排序使用。"""
        if self.client is None or not candidates:
            return {}
        payload = [
            {
                "node_id": candidate.node_id,
                "summary": str(candidate.metadata.get("summary") or ""),
                "keywords": candidate.metadata.get("keywords", []),
                "text": candidate.text[:400],
            }
            for candidate in candidates
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 检索精排器。只返回 JSON："
                        "{\"results\": [{\"node_id\": \"...\", \"score\": 0.0}]}。"
                    ),
                },
                {"role": "user", "content": json.dumps({"query": query, "candidates": payload}, ensure_ascii=False)},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return _parse_rerank_scores(content)


class DualStageReranker:
    """双阶段重排器：先规则粗排，再选择性调用 LLM 对头部候选精排。"""

    def __init__(
        self,
        rule_reranker: RuleBasedReranker | None = None,
        llm_reranker: LLMReranker | None = None,
        llm_enabled: bool = True,
        llm_top_n: int = 10,
    ) -> None:
        self.rule_reranker = rule_reranker or RuleBasedReranker()
        self.llm_reranker = llm_reranker
        self.llm_enabled = llm_enabled
        self.llm_top_n = llm_top_n

    def rerank(self, query: str, candidates: list["RetrievalCandidate"]) -> list["RetrievalCandidate"]:
        """执行两阶段排序；若 LLM 不可用或失败，则自动退回规则粗排结果。"""
        coarse_ranked = self.rule_reranker.rerank(candidates)
        if not self.llm_enabled or self.llm_reranker is None or not coarse_ranked:
            return coarse_ranked

        llm_candidates = coarse_ranked[: self.llm_top_n]
        try:
            llm_scores = self.llm_reranker.rerank(query, llm_candidates)
        except Exception:
            return coarse_ranked

        if not llm_scores:
            return coarse_ranked

        llm_ranked: list["RetrievalCandidate"] = []
        for candidate in llm_candidates:
            if candidate.node_id not in llm_scores:
                continue
            candidate.llm_score = llm_scores[candidate.node_id]
            candidate.final_score = candidate.fused_score * 0.2 + candidate.llm_score * 0.8
            llm_ranked.append(candidate)

        if not llm_ranked:
            return coarse_ranked

        llm_ranked.sort(key=lambda item: item.final_score, reverse=True)
        llm_ranked_ids = {candidate.node_id for candidate in llm_ranked}
        tail = [candidate for candidate in coarse_ranked if candidate.node_id not in llm_ranked_ids]
        return llm_ranked + tail


def _parse_query_rewrite(content: str, limit: int) -> list[str]:
    data = _parse_json_payload(content)
    values = data.get("queries", [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()][:limit]


def _parse_rerank_scores(content: str) -> dict[str, float]:
    data = _parse_json_payload(content)
    results = data.get("results", [])
    if not isinstance(results, list):
        return {}
    scores: dict[str, float] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        try:
            scores[node_id] = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


def _parse_json_payload(content: str) -> dict:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)
