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

from rag.shared.checkpoints import CheckpointStore, ProgressSnapshot, node_key, print_progress


EMPTY_ANNOTATION = {
    "summary": "",
    "keywords": [],
    "tags": [],
    "type": "text",
    "has_code": False,
    "coherence": "medium",
}


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
        prompt_version: str = "annotation_v2",
    ) -> None:
        if not api_key and client is None:
            raise ValueError("缺少 LLM API Key，请设置 DASHSCOPE_API_KEY")
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt or load_annotation_prompt()
        self.prompt_version = prompt_version

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

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.annotate_text, node.text): node for node in pending}
            for future in as_completed(futures):
                node = futures[future]
                key = self._cache_key(node)
                try:
                    annotation = future.result()
                    status = "success"
                    error = None
                except Exception as exc:
                    annotation = dict(EMPTY_ANNOTATION)
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
        logger.info("标注阶段结束 | 总数=%s | 成功=%s | 失败=%s", len(nodes), len(nodes) - failed, failed)
        return AnnotationRunResult(nodes=nodes, failed_keys=failed_keys)

    def annotate_text(self, text: str) -> dict[str, Any]:
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
                return normalize_annotation(parse_json_object(content))
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        if last_error:
            raise last_error
        return dict(EMPTY_ANNOTATION)

    def _cache_key(self, node: Any) -> str:
        return f"{node_key(node)}|annotation|{self.model}|{self.prompt_version}"

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


def normalize_annotation(raw: dict[str, Any]) -> dict[str, Any]:
    annotation = dict(EMPTY_ANNOTATION)
    annotation.update(raw)
    annotation["summary"] = str(annotation.get("summary") or "")[:50]
    annotation["keywords"] = _string_list(annotation.get("keywords"), 6)
    annotation["tags"] = _string_list(annotation.get("tags"), 8)
    annotation["type"] = _enum(annotation.get("type"), {"text", "api", "code", "table"}, "text")
    annotation["has_code"] = bool(annotation.get("has_code"))
    annotation["coherence"] = _enum(annotation.get("coherence"), {"high", "medium", "low"}, "medium")
    return annotation


def load_annotation_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else Path("prompts/annotation_v2.md")
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


DEFAULT_ANNOTATION_PROMPT = """
你是 RAG 离线索引阶段的中文技术文档语义标注器。

任务边界：
- 只分析用户提供的单个 chunk。
- 不改写、不续写、不拆分、不合并原文。
- 不输出解释、Markdown、代码块或额外字段。
- 无法判断时使用保守值，不编造原文不存在的事实。

输出必须是一个严格 JSON 对象，字段如下：
{
  "summary": "不超过 50 个中文字符的可检索摘要",
  "keywords": ["3 到 6 个关键词"],
  "tags": ["1 到 8 个主题标签"],
  "type": "text|api|code|table",
  "has_code": true,
  "coherence": "high|medium|low"
}

标注规则：
- summary 要面向检索，概括 chunk 中可回答的问题、接口、概念、约束或结论。
- keywords 优先选择用户可能搜索的中文词、英文术语、接口名、配置名和同义表达。
- tags 使用稳定的主题或业务域名称，例如 RAG、向量检索、权限、支付、部署、API文档。
- type 判断主内容类型：接口说明为 api，代码占主要内容为 code，表格占主要信息为 table，其余为 text。
- has_code 仅在出现代码块、命令、配置片段、函数签名或 JSON/YAML 示例时为 true。
- coherence 表示该 chunk 独立可理解程度：high 独立完整；medium 需要少量上下文；low 明显残缺、跨页断裂或只有碎片。
""".strip()
