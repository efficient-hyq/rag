from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag.indexing.chunk_context import build_retrieval_text
from rag.indexing.dependency_topics import normalize_dependency_topics
from rag.shared.checkpoints import CheckpointStore, ProgressSnapshot, node_key, print_progress, text_hash


EMPTY_ANNOTATION = {
    "summary": "",
    "keywords": [],
    "tags": [],
    "type": "text",
    "has_code": False,
    "coherence": "medium",
    "dependency_topics": [],
    "subject": {"kind": "其他", "title": "", "identifiers": []},
    "stated_rules": [],
}

SUBJECT_KINDS = {"接口", "配置", "流程", "规则", "错误码", "其他"}
MAX_SUBJECT_IDENTIFIERS = 8
MAX_STATED_RULES = 3


@dataclass(frozen=True)
class AnnotationRunResult:
    nodes: list[Any]
    failed_keys: list[str]

    @property
    def failed_count(self) -> int:
        return len(self.failed_keys)


class SemanticAnnotator:
    """使用 OpenAI 兼容接口为 chunk 写入语义元数据。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.6-plus",
        max_workers: int = 5,
        max_retries: int = 3,
        client: OpenAI | None = None,
        prompt: str | None = None,
        prompt_version: str = "annotation_v5",
    ) -> None:
        if not api_key and client is None:
            raise ValueError("缺少 LLM API Key，请设置 DASHSCOPE_API_KEY")
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt or load_annotation_prompt()
        self.prompt_version = prompt_version
        self.prompt_hash = text_hash(self.prompt)

    def annotate_nodes(
        self,
        nodes: list[Any],
        checkpoint: CheckpointStore | None = None,
        show_progress: bool = True,
    ) -> AnnotationRunResult:
        logger = logging.getLogger("rag.annotator")
        cached = checkpoint.load_annotations() if checkpoint else {}
        failed = 0
        completed = 0
        failed_keys: list[str] = []

        pending: list[Any] = []
        for node in nodes:
            key = self._cache_key(node)
            annotation = cached.get(key)
            if annotation:
                self._apply_annotation(node, annotation)
                completed += 1
            else:
                pending.append(node)

        if show_progress:
            print_progress(ProgressSnapshot("标注", len(nodes), completed, failed))

        start_time = time.time()
        logger.info("开始并发标注 | 待标注=%s | 并发数=%s | 开始时间=%s",
                    len(pending), self.max_workers, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._annotate_with_timing,
                    self._annotation_input(node),
                    str(getattr(node, "text", "") or ""),
                    str((getattr(node, "metadata", {}) or {}).get("heading_path") or ""),
                ): node
                for node in pending
            }
            for future in as_completed(futures):
                node = futures[future]
                key = self._cache_key(node)
                try:
                    annotation, elapsed, start_ts, end_ts = future.result()
                    status = "success"
                    error = None
                    logger.info("标注成功 | node_id=%s | 耗时=%.2fs | 开始=%s | 结束=%s",
                                node.id_, elapsed, start_ts, end_ts)
                except Exception as exc:
                    annotation = normalize_annotation({})
                    status = "failed"
                    error = str(exc)
                    failed += 1
                    failed_keys.append(key)
                    logger.warning("标注失败 | key=%s | error=%s", key, error)

                self._apply_annotation(node, annotation)
                if checkpoint:
                    checkpoint.append_annotation(key, annotation, status=status, error=error)
                completed += 1
                if show_progress:
                    print_progress(ProgressSnapshot("标注", len(nodes), completed, failed))

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info("标注阶段结束 | 总数=%s | 成功=%s | 失败=%s | 耗时=%.2fs | 结束时间=%s",
                    len(nodes), len(nodes) - failed, failed, elapsed,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        return AnnotationRunResult(nodes=nodes, failed_keys=failed_keys)

    def annotate_text(self, text: str, *, source_text: str = "", heading_path: str = "") -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                return normalize_annotation(
                    parse_json_object(content),
                    source_text=source_text or text,
                    heading_path=heading_path,
                )
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        if last_error:
            raise last_error
        return dict(EMPTY_ANNOTATION)

    def _annotate_with_timing(
        self,
        text: str,
        source_text: str,
        heading_path: str,
    ) -> tuple[dict[str, Any], float, str, str]:
        """带计时的标注方法，返回 (annotation, elapsed_time, start_time, end_time)。"""
        start = time.time()
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
        annotation = self.annotate_text(text, source_text=source_text, heading_path=heading_path)
        end = time.time()
        end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
        elapsed = end - start
        return annotation, elapsed, start_ts, end_ts

    def _annotation_input(self, node: Any) -> str:
        """标注读取和正文向量、BM25 相同的检索文本。"""
        return build_retrieval_text(
            str(getattr(node, "text", "") or ""),
            dict(getattr(node, "metadata", {}) or {}),
        )

    def _cache_key(self, node: Any) -> str:
        return (
            f"{node_key(node)}|annotation|{self.prompt_version}|{self.prompt_hash}|"
            f"{text_hash(self._annotation_input(node))}"
        )

    @staticmethod
    def _apply_annotation(node: Any, annotation: dict[str, Any]) -> None:
        metadata = dict(getattr(node, "metadata", {}) or {})
        metadata.update(annotation)
        node.metadata = metadata


def parse_json_object(content: str) -> dict[str, Any]:
    """解析 LLM 输出，兼容 ```json 代码块包裹。"""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def normalize_annotation(
    raw: dict[str, Any],
    *,
    source_text: str = "",
    heading_path: str = "",
) -> dict[str, Any]:
    annotation = dict(EMPTY_ANNOTATION)
    annotation.update(raw)
    annotation["summary"] = str(annotation.get("summary") or "")[:50]
    annotation["keywords"] = _string_list(annotation.get("keywords"), 6)
    annotation["tags"] = _string_list(annotation.get("tags"), 8)
    annotation["type"] = _enum(annotation.get("type"), {"text", "api", "code", "table"}, "text")
    annotation["has_code"] = bool(annotation.get("has_code"))
    annotation["coherence"] = _enum(annotation.get("coherence"), {"high", "medium", "low"}, "medium")
    annotation["dependency_topics"] = normalize_dependency_topics(annotation.get("dependency_topics"))
    annotation["subject"] = _normalize_subject(raw.get("subject"), source_text, heading_path)
    annotation["stated_rules"] = _normalize_stated_rules(raw.get("stated_rules"), source_text)
    return annotation


def load_annotation_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else Path("prompts/annotation_v5.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_ANNOTATION_PROMPT


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or default).strip().lower()
    return text if text in allowed else default


def _normalize_subject(raw: Any, source_text: str, heading_path: str) -> dict[str, Any]:
    """整理当前段身份。具体名字必须能在正文或标题里原样找到。"""
    payload = raw if isinstance(raw, dict) else {}
    kind = str(payload.get("kind") or "其他").strip()
    if kind not in SUBJECT_KINDS:
        kind = "其他"
    return {
        "kind": kind,
        "title": str(payload.get("title") or "").strip()[:80],
        "identifiers": _grounded_identifiers(
            payload.get("identifiers"),
            f"{source_text}\n{heading_path}",
            MAX_SUBJECT_IDENTIFIERS,
        ),
    }


def _normalize_stated_rules(raw: Any, source_text: str = "") -> list[dict[str, Any]]:
    """整理规矩的结构。要求和依据允许概括，不再因为没有逐字出现在正文里就丢弃。"""
    del source_text
    if not isinstance(raw, list):
        return []
    rules: list[dict[str, Any]] = []
    for item in raw:
        if len(rules) >= MAX_STATED_RULES or not isinstance(item, dict):
            continue
        evidence = _compact_text(item.get("evidence"))
        requirement = _compact_text(item.get("requirement"))
        applies_to = _compact_text(item.get("applies_to"))
        if not evidence or not requirement or not applies_to:
            continue
        rules.append(
            {
                "requirement": requirement[:80],
                "applies_to": applies_to[:120],
                "required": _coerce_required(item.get("required")),
                "identifiers": _clean_identifiers(item.get("identifiers"), MAX_SUBJECT_IDENTIFIERS),
                "evidence": evidence[:200],
            }
        )
    return rules


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _coerce_required(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return str(value or "").strip().casefold() in {"true", "1", "yes", "是", "必须", "需要", "强制"}


def _clean_identifiers(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip()
        key = token.casefold()
        if token and key not in seen:
            result.append(token[:80])
            seen.add(key)
            if len(result) >= limit:
                break
    return result


def _grounded_identifiers(value: Any, source: str, limit: int) -> list[str]:
    if not isinstance(value, list) or not source.strip():
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip()
        key = token.casefold()
        if token and token in source and key not in seen:
            result.append(token[:80])
            seen.add(key)
            if len(result) >= limit:
                break
    return result


DEFAULT_ANNOTATION_PROMPT = """
你是 RAG 离线索引阶段的中文技术文档语义标注器。

任务边界：
- 只分析用户提供的单个 chunk。
- 不改写、不续写、不拆分、不合并原文。
- 不输出解释、Markdown、代码块或额外字段。
- 无法判断时使用保守值。摘要可以吸收定位说明里的位置和主体，但不能编造定位说明和正文都没有的事实。

输出必须是一个严格 JSON 对象，字段如下：
{
  "summary": "不超过 50 个中文字符的可检索摘要",
  "keywords": ["3 到 6 个关键词"],
  "tags": ["1 到 8 个主题标签"],
  "type": "text|api|code|table",
  "has_code": true,
  "coherence": "high|medium|low",
  "dependency_topics": ["LLM 自由概括的简短依赖主题，最多 5 个"],
  "subject": {"kind": "接口|配置|流程|规则|错误码|其他", "title": "这段自己的标题", "identifiers": []},
  "stated_rules": []
}

标注规则：
- 用户消息是送去检索的文本。归档字段之前的那段话是定位说明。
- summary 要面向检索，概括检索文本可回答的问题、接口、概念、约束或结论，并带上定位说明中的位置和主体。
- keywords 优先选择用户可能搜索的中文词、英文术语、接口名、配置名和同义表达。
- tags 使用稳定的主题或业务域名称，例如 RAG、向量检索、权限、支付、部署、API文档。
- type 判断主内容类型：接口说明为 api，代码占主要内容为 code，表格占主要信息为 table，其余为 text。
- has_code 仅在出现代码块、命令、配置片段、函数签名或 JSON/YAML 示例时为 true。
- coherence 表示该 chunk 独立可理解程度：high 独立完整；medium 需要少量上下文；low 明显残缺、跨页断裂或只有碎片。
- dependency_topics 只依据正文自由概括为简短、可读的中文主题；不受固定词表约束，正文没有依据时返回空数组，不根据定位说明或通用常识补充，最多 5 个。
- dependency_topics 表示当前 chunk 正文提供的依赖能力证据，不表示页面类型或依赖关系类型；仅提及名词但未提供规则、配置、约束或操作证据时不要标注。
- 通常返回 0 到 3 个 topic，优先使用正文中的稳定业务术语。
- subject 描述当前 chunk 自己是什么。kind 只能取接口、配置、流程、规则、错误码、其他。identifiers 必须是正文或标题中的连续原文。
- stated_rules 只在正文替别的接口、配置、回调或请求定下规矩时填写。requirement 和 applies_to 用短词概括，evidence 保留一句依据，都不必逐字照抄。本段没有约束别人时返回空数组。
- 不要输出 chunk 之间的依赖边。
""".strip()
