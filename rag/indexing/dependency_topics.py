"""依赖主题的宽松清洗与依赖关系字段规范化。"""

from __future__ import annotations

from typing import Any


RELATION_TYPES = {
    "prerequisite",
    "constraint",
    "failure_handling",
    "follow_up",
}


def normalize_dependency_topics(value: Any, limit: int = 5) -> list[str]:
    """清洗 LLM 自由标注的 topic，并去重保持原顺序。"""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        topic = str(item or "").strip()[:50]
        dedupe_key = topic.casefold()
        if topic and dedupe_key not in seen:
            result.append(topic)
            seen.add(dedupe_key)
            if len(result) >= limit:
                break
    return result


def normalize_relation_edge(raw: Any) -> dict[str, Any] | None:
    """规范化单条依赖边；topic 仅为可选说明，不参与有效性判断。"""
    if not isinstance(raw, dict):
        return None
    target = str(raw.get("node_id") or "").strip()
    relation = str(raw.get("relation_type") or "").strip().lower()
    reason = str(raw.get("reason") or "").strip()
    required = raw.get("required")
    raw_confidence = raw.get("confidence")
    if isinstance(raw_confidence, bool):
        return None
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return None
    if (
        not target
        or relation not in RELATION_TYPES
        or not isinstance(required, bool)
        or not 0 <= confidence <= 1
        or not reason
    ):
        return None

    edge = {
        "node_id": target,
        "relation_type": relation,
        "required": required,
        "confidence": confidence,
        "reason": reason[:200],
    }
    topic = str(
        raw.get("topic")
        or raw.get("topic_name")
        or raw.get("topic_code")
        or ""
    ).strip()[:50]
    if topic:
        edge["topic"] = topic
    return edge
