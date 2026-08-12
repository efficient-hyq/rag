"""在线隐式依赖路由、seed 选择与一层图扩展。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:
    from rag.retrieval.retriever import MetadataIndex, RetrievalCandidate


@dataclass(frozen=True)
class DependencyRecallDecision:
    need_dependency_recall: bool
    intent_type: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class DependencyExpansionResult:
    candidates: list["RetrievalCandidate"]
    seeds: list["RetrievalCandidate"]
    dependency_by_source: dict[str, list[str]]
    added_count: int
    missing_count: int
    truncated_count: int


class DependencyGraphIndex:
    """只加载通过完整性校验的文档依赖图。"""

    def __init__(self, edges_by_source: dict[str, list[dict[str, Any]]]) -> None:
        self.edges_by_source = edges_by_source

    @classmethod
    def from_file(cls, path: str | Path) -> "DependencyGraphIndex":
        cards_path = Path(path)
        if not cards_path.exists():
            return cls({})
        payload = json.loads(cards_path.read_text(encoding="utf-8"))
        documents = payload.get("documents") if isinstance(payload, dict) else None
        if not isinstance(documents, dict):
            raise ValueError("dependency_cards.json 格式非法")
        edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for document in documents.values():
            if not _is_publishable_document(document):
                continue
            for source_id, edges in document["node_dependencies"].items():
                if isinstance(edges, list):
                    edges_by_source[str(source_id)] = [dict(edge) for edge in edges if isinstance(edge, dict)]
        return cls(edges_by_source)

    def get_required_edges(
        self,
        source_node_id: str,
        min_confidence: float = 0.7,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        for edge in self.edges_by_source.get(source_node_id, []):
            try:
                confidence = float(edge.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            if edge.get("required") is not True or confidence < min_confidence:
                continue
            item = dict(edge)
            item["confidence"] = confidence
            valid.append(item)
        valid.sort(key=lambda item: item["confidence"], reverse=True)
        return valid[: max(limit, 0)]


class DependencyRecallRouter:
    """优先用确定性规则判断，边界问题可交给 LLM 分类。"""

    POSITIVE_TERMS = (
        "怎么接入",
        "如何接入",
        "怎么实现",
        "如何实现",
        "完整流程",
        "哪些步骤",
        "上线前",
        "如何配置",
        "怎么配置",
        "如何排查",
        "怎么排查",
        "需要注意",
    )
    LOOKUP_TERMS = ("是什么", "表示什么", "返回哪些字段", "有哪些字段", "字段含义", "错误码")

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "qwen3.6-plus",
        llm_enabled: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.llm_enabled = llm_enabled

    def decide(self, query: str) -> DependencyRecallDecision:
        normalized = query.strip().lower()
        if any(term in normalized for term in self.POSITIVE_TERMS):
            return DependencyRecallDecision(True, "procedure", 1.0, "命中流程类规则")
        if any(term in normalized for term in self.LOOKUP_TERMS):
            return DependencyRecallDecision(False, "lookup", 1.0, "命中事实查询规则")
        if not self.llm_enabled or self.client is None:
            return DependencyRecallDecision(False, "lookup", 0.5, "规则未命中且未启用路由模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "判断问题是否需要补充业务前置、约束或异常处理依赖。"
                            "只返回 JSON：{\"need_dependency_recall\":true,"
                            "\"intent_type\":\"integration|procedure|lookup|troubleshooting|configuration\","
                            "\"confidence\":0.0}。不要决定具体依赖 node。"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0,
            )
            payload = _parse_json_object(response.choices[0].message.content or "")
            return DependencyRecallDecision(
                payload.get("need_dependency_recall") is True,
                str(payload.get("intent_type") or "lookup"),
                max(0.0, min(1.0, float(payload.get("confidence", 0)))),
                "路由模型判断",
            )
        except Exception as exc:
            logging.getLogger("rag.dependency_recall").warning("依赖路由判断失败 | error=%s", exc)
            return DependencyRecallDecision(False, "lookup", 0.0, "路由模型失败，保守关闭")


def calculate_dependency_capability_score(candidate: "RetrievalCandidate") -> float:
    """评估候选chunk作为依赖seed的潜力，识别并降权干扰文档。"""
    score = 0.0
    metadata = candidate.metadata

    # 1. dependency_topics 非空表示有依赖能力证据，+0.3
    topics = metadata.get("dependency_topics", [])
    if topics:
        score += 0.3

    # 2. heading_path 包含前置规范关键词，+0.4
    heading_path = str(metadata.get("heading_path", ""))
    prerequisite_keywords = ["说明", "规范", "配置", "流程", "机制", "错误码", "认证", "签名", "加密", "鉴权", "初始化"]
    if any(kw in heading_path for kw in prerequisite_keywords):
        score += 0.4

    # 3. chunk 类型是 "api"（接口定义），+0.2
    if metadata.get("type") == "api":
        score += 0.2

    # 4. heading_path 包含通用词，识别为干扰文档，-0.3
    generic_keywords = ["介绍", "概述", "简介", "背景", "产品", "优势", "特性"]
    if any(kw in heading_path for kw in generic_keywords):
        score -= 0.3

    # 5. 质量评分低于阈值，-0.2
    quality_score = metadata.get("quality_score", 1.0)
    if quality_score < 0.7:
        score -= 0.2

    return max(0.0, min(1.0, score))


def select_dependency_seeds(
    candidates: list["RetrievalCandidate"],
    score_ratio: float = 0.7,
    max_seeds: int = 8,
    capability_weight: float = 0.3,
) -> list["RetrievalCandidate"]:
    """选择依赖扩展的seed，混合final_score和依赖能力评分。"""
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.is_neighbor
        and not any(hit.route in {"neighbor_expand", "dependency_graph"} for hit in candidate.hits)
    ]
    if not eligible or max_seeds <= 0:
        return []

    # 计算混合分数：final_score * (1-w) + capability_score * w
    for candidate in eligible:
        capability_score = calculate_dependency_capability_score(candidate)
        candidate.dependency_seed_score = (
            candidate.final_score * (1 - capability_weight) + capability_score * capability_weight
        )

    # 按混合分数重新排序
    eligible_sorted = sorted(eligible, key=lambda c: getattr(c, "dependency_seed_score", c.final_score), reverse=True)

    # 选择seeds
    highest = getattr(eligible_sorted[0], "dependency_seed_score", eligible_sorted[0].final_score)
    threshold = highest * max(0.0, score_ratio)
    seeds: list["RetrievalCandidate"] = []
    for candidate in eligible_sorted:
        seed_score = getattr(candidate, "dependency_seed_score", candidate.final_score)
        if seeds and seed_score < threshold:
            break
        seeds.append(candidate)
        if len(seeds) >= max_seeds:
            break
    return seeds


def expand_dependency_candidates(
    query: str,
    candidates: list["RetrievalCandidate"],
    seeds: list["RetrievalCandidate"],
    graph: DependencyGraphIndex,
    metadata_index: "MetadataIndex",
    min_confidence: float = 0.7,
    per_seed_limit: int = 5,
    total_limit: int = 12,
) -> DependencyExpansionResult:
    from rag.retrieval.retriever import RetrievalCandidate, RetrievedRouteHit

    logger = logging.getLogger("rag.dependency_recall")
    expanded = list(candidates)
    by_id = {candidate.node_id: candidate for candidate in expanded}
    dependency_by_source: dict[str, list[str]] = {seed.node_id: [] for seed in seeds}
    selected_context_ids = {seed.node_id for seed in seeds}
    added_count = 0
    missing_count = 0
    truncated_count = 0

    pending: list[tuple[float, int, RetrievalCandidate, dict[str, Any], str]] = []
    for seed_rank, seed in enumerate(seeds, start=1):
        for edge in graph.get_required_edges(seed.node_id, min_confidence, per_seed_limit):
            target_id = str(edge.get("node_id") or "")
            resolved_id = target_id
            if resolved_id not in metadata_index.metadata_by_id:
                fallback_id = metadata_index.find_dependency_fallback(seed.node_id, edge)
                if fallback_id is None:
                    missing_count += 1
                    logger.warning(
                        "依赖证据缺失 | source_node_id=%s | target_node_id=%s",
                        seed.node_id,
                        target_id,
                    )
                    continue
                resolved_id = fallback_id
                logger.warning(
                    "依赖证据使用同文档兜底 | source_node_id=%s | missing_target=%s | fallback=%s",
                    seed.node_id,
                    target_id,
                    fallback_id,
                )
            pending.append((float(edge["confidence"]), seed_rank, seed, edge, resolved_id))

    pending.sort(key=lambda item: (-item[0], item[1]))
    for confidence, seed_rank, seed, edge, resolved_id in pending:
        is_new_context = resolved_id not in selected_context_ids
        if is_new_context and len(selected_context_ids) >= total_limit:
            truncated_count += 1
            logger.warning(
                "必要依赖因安全上限被裁剪 | source_node_id=%s | target_node_id=%s | confidence=%.3f",
                seed.node_id,
                resolved_id,
                confidence,
            )
            continue
        candidate = by_id.get(resolved_id)
        metadata = dict(metadata_index.metadata_by_id.get(resolved_id, {}))
        if candidate is None:
            candidate = RetrievalCandidate(
                node_id=resolved_id,
                text=str(metadata.get("text") or ""),
                metadata=metadata,
            )
            expanded.append(candidate)
            by_id[resolved_id] = candidate
            added_count += 1
        if not any(hit.route == "dependency_graph" and hit.query_text == seed.node_id for hit in candidate.hits):
            candidate.hits.append(
                RetrievedRouteHit(
                    node_id=resolved_id,
                    route="dependency_graph",
                    rank=seed_rank,
                    raw_score=confidence,
                    query_text=seed.node_id,
                    is_rewritten=False,
                )
            )
        candidate.is_dependency_evidence = True
        source_info = {
            "source_node_id": seed.node_id,
            "relation_type": str(edge.get("relation_type") or ""),
            "required": True,
            "confidence": confidence,
            "reason": str(edge.get("reason") or ""),
        }
        topic = str(edge.get("topic") or "").strip()
        if topic:
            source_info["topic"] = topic
        if source_info not in candidate.dependency_sources:
            candidate.dependency_sources.append(source_info)
        dependency_by_source[seed.node_id].append(resolved_id)
        selected_context_ids.add(resolved_id)

    return DependencyExpansionResult(
        candidates=expanded,
        seeds=seeds,
        dependency_by_source=dependency_by_source,
        added_count=added_count,
        missing_count=missing_count,
        truncated_count=truncated_count,
    )


def assemble_dependency_context(
    seeds: list["RetrievalCandidate"],
    dependency_by_source: dict[str, list[str]],
    candidates: list["RetrievalCandidate"],
    final_top_n: int = 5,
    max_forced_dependencies: int = 15,
    ensure_seed_in_context: bool = True,
) -> list["RetrievalCandidate"]:
    """合并常规 TopN 与按 seed 顺序排列的强制依赖证据。

    依赖扩展阶段已经通过 ``total_limit`` 防止依赖证据无限增长，因此此处不再
    对它们截断，也不让它们挤占常规重排 TopN 的配额。

    Args:
        seeds: 依赖扩展的种子候选
        dependency_by_source: seed到依赖node_id的映射
        candidates: 重排后的所有候选
        final_top_n: 常规TopN数量
        max_forced_dependencies: 强制依赖证据的最大数量
        ensure_seed_in_context: 确保高分seed进入上下文
    """
    logger = logging.getLogger("rag.dependency_recall")
    by_id = {candidate.node_id: candidate for candidate in candidates}
    result: list["RetrievalCandidate"] = []
    seen: set[str] = set()

    # 第一步：确保高分seeds进入上下文（如果它们在TopN范围内）
    if ensure_seed_in_context:
        for seed in seeds[: min(final_top_n, len(seeds))]:
            if seed.node_id in by_id:
                result.append(seed)
                seen.add(seed.node_id)

    # 第二步：补充常规TopN（去重）
    for candidate in candidates[: max(final_top_n, 0)]:
        if candidate.node_id not in seen:
            result.append(candidate)
            seen.add(candidate.node_id)

    # 第三步：追加强制依赖证据（按seed顺序，受max_forced_dependencies限制）
    forced_count = 0
    for seed in seeds:
        for dependency_id in dependency_by_source.get(seed.node_id, []):
            if dependency_id in seen or dependency_id not in by_id:
                continue
            if forced_count >= max_forced_dependencies:
                logger.warning(
                    "强制依赖证据达到上限 | max_forced_dependencies=%s | 停止追加",
                    max_forced_dependencies,
                )
                break
            result.append(by_id[dependency_id])
            seen.add(dependency_id)
            forced_count += 1
        if forced_count >= max_forced_dependencies:
            break

    return result


def _is_publishable_document(document: Any) -> bool:
    if not isinstance(document, dict) or document.get("status") != "success":
        return False
    dependencies = document.get("node_dependencies")
    return (
        isinstance(dependencies, dict)
        and document.get("expected_node_count") == document.get("processed_node_count")
        and document.get("expected_node_count") == len(dependencies)
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("依赖路由输出必须是 JSON 对象")
    return payload
