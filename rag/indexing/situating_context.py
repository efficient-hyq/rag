"""为每个 chunk 生成只用于检索的定位上下文。

整篇文档和当前片段一起交给模型，得到 50 到 100 token 的定位说明。
说明写在 metadata 里，不改 chunk 原文。正文向量和 BM25 会把它接在检索文本前面。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag.config.common import resolve_project_path
from rag.indexing.storage_indexer import doc_key_from_metadata
from rag.shared.checkpoints import CheckpointStore, ProgressSnapshot, node_key, print_progress, text_hash


logger = logging.getLogger("rag.indexing.situating")

SITUATING_PROMPT_VERSION = "chunk_context_v1"
_MAX_CONTEXT_CHARS = 400


def load_situating_prompt(path: str | Path) -> str:
    """读取定位提示词。相对路径按项目根目录解析。"""
    prompt_path = resolve_project_path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"未找到片段定位提示词: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def render_situating_prompt(template: str, document: str, chunk: str) -> str:
    """把整篇文档和当前片段填进提示词。文档在前，便于同一篇文档复用前缀。"""
    rendered = str(template or "")
    rendered = rendered.replace("{{WHOLE_DOCUMENT}}", document)
    rendered = rendered.replace("{{CHUNK_CONTENT}}", chunk)
    return rendered


def load_whole_documents(nodes: list[Any], docs_dir: str | Path) -> dict[str, str]:
    """按文档读取全文。找不到预处理 Markdown 时，按 chunk 顺序拼回。"""
    root = Path(docs_dir)
    grouped: dict[str, list[Any]] = {}
    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_metadata(metadata, root)
        grouped.setdefault(doc_key, []).append(node)

    documents: dict[str, str] = {}
    for doc_key, doc_nodes in grouped.items():
        relative = str((getattr(doc_nodes[0], "metadata", {}) or {}).get("markdown_relative_path") or "")
        file_path = root / relative if relative else None
        if file_path is not None and file_path.is_file():
            documents[doc_key] = file_path.read_text(encoding="utf-8")
            continue
        ordered = sorted(doc_nodes, key=lambda item: int((getattr(item, "metadata", {}) or {}).get("chunk_index") or 0))
        documents[doc_key] = "\n\n".join(str(getattr(item, "text", "") or "") for item in ordered)
    return documents


def situating_cache_key(node_id: str, prompt_hash: str, document_hash: str, chunk_hash: str) -> str:
    return f"{node_id}|situating|{prompt_hash}|{document_hash}|{chunk_hash}"


def clean_situating_context(raw: str) -> str:
    """只保留模型给出的一段定位说明。"""
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = " ".join(text.split())
    return text[:_MAX_CONTEXT_CHARS]


class SituatingContextGenerator:
    """调用对话模型，为每个片段写定位上下文。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        prompt: str,
        prompt_hash: str,
        max_workers: int = 4,
        max_retries: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("缺少 LLM API Key，请设置 DASHSCOPE_API_KEY")
        if not str(model or "").strip():
            raise ValueError("缺少片段定位模型，请设置 RAG_CHUNK_CONTEXT_MODEL")
        if "{{WHOLE_DOCUMENT}}" not in prompt or "{{CHUNK_CONTENT}}" not in prompt:
            raise ValueError("片段定位提示词必须包含 {{WHOLE_DOCUMENT}} 和 {{CHUNK_CONTENT}}")
        self.model = str(model).strip()
        self.prompt = prompt
        self.prompt_hash = prompt_hash
        self.max_workers = max(int(max_workers), 1)
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def apply(
        self,
        nodes: list[Any],
        documents: dict[str, str],
        document_hashes: dict[str, str],
        checkpoint: CheckpointStore | None = None,
    ) -> None:
        """把定位上下文写进节点 metadata。已有检查点的片段不再请求模型。"""
        cached = checkpoint.load_situating_contexts() if checkpoint else {}
        pending: list[Any] = []
        completed = 0
        for node in nodes:
            key = self._cache_key(node, documents, document_hashes)
            context = cached.get(key, "")
            if context:
                _store_situating_context(node, context)
                completed += 1
            else:
                pending.append(node)

        print_progress(ProgressSnapshot("定位上下文", len(nodes), completed, 0))
        if not pending:
            logger.info("定位上下文已全部命中检查点 | 总数=%s", len(nodes))
            return

        logger.info("开始生成定位上下文 | 待生成=%s | 并发数=%s", len(pending), self.max_workers)
        failed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._generate_for_node, node, documents): node for node in pending}
            for future in as_completed(futures):
                node = futures[future]
                key = self._cache_key(node, documents, document_hashes)
                try:
                    context = future.result()
                    status = "success"
                    error = None
                except Exception as exc:
                    context = ""
                    status = "failed"
                    error = str(exc)
                    failed += 1
                    logger.warning("定位上下文失败 | key=%s | error=%s", key, error)
                if context:
                    _store_situating_context(node, context)
                if checkpoint:
                    checkpoint.append_situating_context(key, context, status=status, error=error)
                completed += 1
                print_progress(ProgressSnapshot("定位上下文", len(nodes), completed, failed))

        if failed:
            raise RuntimeError(f"定位上下文存在 {failed} 个失败 chunk，不会继续标注和入库。")

    def _generate_for_node(self, node: Any, documents: dict[str, str]) -> str:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_metadata(metadata, Path("."))
        document = documents.get(doc_key) or str(getattr(node, "text", "") or "")
        chunk = str(getattr(node, "text", "") or "")
        prompt = render_situating_prompt(self.prompt, document, chunk)
        return self._request(prompt)

    def _request(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "只输出用于检索的简短定位上下文，不要解释。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                content = clean_situating_context(response.choices[0].message.content or "")
                if not content:
                    raise RuntimeError("模型没有返回定位上下文")
                return content
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        if last_error:
            raise last_error
        raise RuntimeError("模型没有返回定位上下文")

    def _cache_key(self, node: Any, documents: dict[str, str], document_hashes: dict[str, str]) -> str:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_key = doc_key_from_metadata(metadata, Path("."))
        document_hash = document_hashes.get(doc_key) or text_hash(documents.get(doc_key, ""))
        chunk_hash = text_hash(str(getattr(node, "text", "") or ""))
        return situating_cache_key(node_key(node), self.prompt_hash, document_hash, chunk_hash)


def _store_situating_context(node: Any, context: str) -> None:
    metadata = dict(getattr(node, "metadata", {}) or {})
    metadata["situating_context"] = context
    node.metadata = metadata
