"""已切分 chunk 的路径、上下文文本和内容指纹。

切分阶段只保留预处理后的文档相对路径。后续标注、上下文嵌入、
关键词索引和重排都从这份路径取出归档目录，避免不同业务线只因正文用词相近而被召回。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

from rag.retrieval.tokenization import estimate_token_size, tokenize_technical_text
from rag.shared.checkpoints import stable_chunk_id, text_hash


# 改变上下文文本的拼法时递增。内容指纹包含该版本，旧 chunk 会重新标注和嵌入。
CONTEXT_VERSION = "archive_v1"

# 这些词在多个业务目录里都会出现，不能用来判断问题属于哪一条归档路径。
GENERIC_ARCHIVE_TOKENS = frozenset(
    {
        "支付",
        "订阅",
        "服务",
        "文档",
        "说明",
        "指引",
        "配置",
        "接口",
        "产品",
        "接入",
        "流程",
        "介绍",
        "概述",
        "业务",
    }
)

# 目录名常用英文，提问常用中文。只在目录本身命中某一组时补上另一写法。
ARCHIVE_ALIAS_GROUPS = (
    frozenset({"google", "谷歌"}),
    frozenset({"apple", "苹果"}),
    frozenset({"wechat", "微信"}),
    frozenset({"alipay", "支付宝"}),
    frozenset({"huawei", "华为"}),
)

DROPPED_METADATA_KEYS = (
    "source_doc_id",
    "source_path",
    "source_absolute_path",
    "source_relative_path",
    "source_file_name",
    "source_file_type",
    "converted_html_path",
    "cleaned_markdown_path",
    "cleaned_markdown_relative_path",
    "file_path",
    "filename",
    "file_size",
    "creation_date",
    "last_modified_date",
)


def archive_path_from_relative(relative: str) -> str:
    """从预处理文档相对路径取出目录，不含文件名。"""
    normalized = str(relative or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    parent = Path(normalized).parent.as_posix()
    if parent in {"", "."}:
        return ""
    return parent


def archive_leaf(archive_path: str) -> str:
    """归档路径的最后一级目录，用来区分同级业务线。"""
    parts = [part.strip() for part in str(archive_path or "").replace("\\", "/").split("/") if part.strip() and part != "."]
    return parts[-1] if parts else ""


def archive_alias_text(archive_path: str) -> str:
    """返回目录名里没有写出、但用户会用来提问的别名。"""
    leaf = archive_leaf(archive_path)
    if not leaf:
        return ""
    raw = leaf.casefold()
    tokens = set(tokenize_technical_text(leaf))
    extras: list[str] = []
    for group in ARCHIVE_ALIAS_GROUPS:
        if not _group_matches(group, tokens, raw):
            continue
        for alias in sorted(group):
            if alias in raw or alias in tokens or alias in extras:
                continue
            extras.append(alias)
    return "、".join(extras)


def archive_match_terms(archive_path: str) -> set[str]:
    """目录中能区分业务线的词，已展开常见中英文别名。"""
    leaf = archive_leaf(archive_path)
    if not leaf:
        return set()
    terms = _expand_aliases(set(tokenize_technical_text(leaf)), leaf)
    return {term for term in terms if term not in GENERIC_ARCHIVE_TOKENS and len(term) >= 2}


def query_terms_for_archive(query: str) -> set[str]:
    """从问题里取出可以和归档目录对照的词。"""
    text = str(query or "").strip()
    if not text:
        return set()
    terms = _expand_aliases(set(tokenize_technical_text(text)), text)
    return {term for term in terms if term not in GENERIC_ARCHIVE_TOKENS and len(term) >= 2}


def build_context_text(text: str, metadata: dict[str, Any] | None) -> str:
    """拼出归档路径、章节和正文。

    标注、正文向量和 BM25 使用 build_retrieval_text，那里会再接上定位上下文。

    返回值不替换 chunk 原文。答案和引用仍读取原来的正文。
    """
    payload = dict(metadata or {})
    archive = str(payload.get("archive_path") or "")
    if not archive:
        archive = archive_path_from_relative(
            str(payload.get("markdown_relative_path") or payload.get("doc_id") or "")
        )
    file_name = str(payload.get("file_name") or "")
    heading = str(payload.get("heading_path") or "")
    lines: list[str] = []
    if archive:
        lines.append(f"归档：{archive}")
        alias = archive_alias_text(archive)
        if alias:
            lines.append(f"别名：{alias}")
    if file_name:
        lines.append(f"文档：{file_name}")
    if heading:
        lines.append(f"章节：{heading}")
    lines.append("正文：")
    lines.append(str(text or ""))
    return "\n".join(lines)


def build_retrieval_text(text: str, metadata: dict[str, Any] | None) -> str:
    """拼出送去正文向量和 BM25 的文本。

    定位上下文在最前，后面仍保留归档路径和原文。chunk 原文本身不修改。
    标注、正文向量和 BM25 使用同一份文本。
    """
    situating = " ".join(str((metadata or {}).get("situating_context") or "").split())
    base = build_context_text(text, metadata)
    if not situating:
        return base
    return f"{situating}\n\n{base}"


def chunk_content_hash(record: dict[str, Any]) -> str:
    """chunk 替换判断用的指纹，覆盖正文、标题、归档路径和上下文版本。"""
    payload = {
        "context_version": CONTEXT_VERSION,
        "text": str(record.get("text") or ""),
        "heading_path": str(record.get("heading_path") or ""),
        "markdown_relative_path": str(record.get("markdown_relative_path") or ""),
        "archive_path": str(record.get("archive_path") or ""),
        "chunk_index": int(record.get("chunk_index") or 0),
        "images": record.get("images") if isinstance(record.get("images"), list) else [],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text_hash(encoded)


def document_content_hash(chunk_hashes: list[str], annotation_prompt_hash: str) -> str:
    """文档指纹由所属 chunk 指纹和标注协议组成，供依赖卡片判断是否失效。"""
    body = annotation_prompt_hash + "\n" + "\n".join(sorted(chunk_hashes))
    return text_hash(body)


def normalize_chunk_record(raw: dict[str, Any]) -> dict[str, Any]:
    """整理一条已切分 chunk，去掉原始文档路径，只留预处理后的相对路径。"""
    if not isinstance(raw, dict):
        raise ValueError("chunk 记录必须是对象")
    relative = str(raw.get("markdown_relative_path") or raw.get("doc_id") or "").replace("\\", "/").strip()
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"chunk 缺少预处理文档相对路径 markdown_relative_path: {relative}")
    text = str(raw.get("text") or "")
    chunk_index = int(raw.get("chunk_index") or 0)
    archive = str(raw.get("archive_path") or "").replace("\\", "/").strip() or archive_path_from_relative(relative)
    file_name = str(raw.get("file_name") or Path(relative).name)
    node_id = str(raw.get("node_id") or "").strip() or stable_chunk_id(relative, chunk_index, text)
    images = raw.get("images") if isinstance(raw.get("images"), list) else []
    token_size = raw.get("token_size")
    if isinstance(token_size, bool) or not isinstance(token_size, int):
        token_size = estimate_token_size(text)
    return {
        "node_id": node_id,
        "doc_id": relative,
        "text": text,
        "heading_path": str(raw.get("heading_path") or ""),
        "chunk_index": chunk_index,
        "token_size": token_size,
        "file_name": file_name,
        "markdown_relative_path": relative,
        "archive_path": archive,
        "images": images,
    }


def load_prepared_chunk_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 prepare_chunks 导出的 jsonl。"""
    chunks_file = Path(path)
    if not chunks_file.is_file():
        raise FileNotFoundError(
            f"未找到已切分的 chunk 文件: {chunks_file}。请先运行 python -m rag.cli.prepare_chunks"
        )
    records: list[dict[str, Any]] = []
    with chunks_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"chunk 文件第 {line_number} 行不是合法 JSON: {chunks_file}") from exc
            records.append(normalize_chunk_record(payload))
    if not records:
        raise RuntimeError(f"切分文件没有任何 chunk: {chunks_file}")
    node_ids = [record["node_id"] for record in records]
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError(f"切分文件存在重复 node_id: {chunks_file}")
    return records


def records_to_nodes(records: list[dict[str, Any]]) -> list[TextNode]:
    """把已整理的 chunk 记录还原成后续标注和入库使用的节点。"""
    nodes: list[TextNode] = []
    for record in records:
        metadata = {
            "doc_id": record["doc_id"],
            "file_name": record["file_name"],
            "file_type": "md",
            "markdown_relative_path": record["markdown_relative_path"],
            "archive_path": record["archive_path"],
            "heading_path": record["heading_path"],
            "chunk_index": record["chunk_index"],
            "token_size": record["token_size"],
            "images": record["images"],
        }
        nodes.append(TextNode(id_=record["node_id"], text=record["text"], metadata=metadata))
    return nodes


def _group_matches(group: frozenset[str], tokens: set[str], raw: str) -> bool:
    if tokens & group:
        return True
    return any(alias in raw for alias in group)


def _expand_aliases(tokens: set[str], raw: str) -> set[str]:
    expanded = {token for token in tokens if token}
    raw_cf = raw.casefold()
    for group in ARCHIVE_ALIAS_GROUPS:
        if _group_matches(group, expanded, raw_cf):
            expanded.update(group)
    return expanded
