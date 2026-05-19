from __future__ import annotations

import re
from typing import Iterable

from llama_index.core.schema import Document, TextNode

from rag.retrieval.tokenization import estimate_token_size
from rag.shared.checkpoints import stable_chunk_id


MARKDOWN_TABLE_LINE = re.compile(r"^\s*\|.+\|\s*$")
MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def split_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[TextNode]:
    """按 Markdown 语义块切分文档，并追加 chunk 级元数据。"""
    nodes: list[TextNode] = []
    for document in documents:
        metadata = dict(document.metadata or {})
        doc_id = str(
            metadata.get("doc_id")
            or metadata.get("file_path")
            or getattr(document, "doc_id", None)
            or getattr(document, "id_", "unknown")
        )
        chunks = split_markdown_text(document.get_content(), chunk_size, chunk_overlap)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["token_size"] = estimate_token_size(chunk_text)
            nodes.append(
                TextNode(
                    id_=stable_chunk_id(doc_id, chunk_index, chunk_text),
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
    return nodes


def split_markdown_text(markdown: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按 Markdown 语义块切分，尽量避免拆断表格和代码块。"""
    blocks = list(split_markdown_blocks(markdown))
    chunks: list[str] = []
    current_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = estimate_token_size(block)
        if current_blocks and current_tokens + block_tokens > chunk_size:
            chunks.append("\n\n".join(current_blocks).strip())
            overlap_blocks = _pick_overlap_blocks(current_blocks, chunk_overlap)
            current_blocks = overlap_blocks[:]
            current_tokens = estimate_token_size("\n\n".join(current_blocks)) if current_blocks else 0

        if block_tokens > chunk_size and not _is_atomic_markdown_block(block):
            for part in _split_large_text_block(block, chunk_size):
                if current_blocks:
                    chunks.append("\n\n".join(current_blocks).strip())
                    current_blocks = []
                    current_tokens = 0
                chunks.append(part.strip())
            continue

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        chunks.append("\n\n".join(current_blocks).strip())
    return [chunk for chunk in chunks if chunk.strip()]


def split_markdown_blocks(markdown: str) -> Iterable[str]:
    """将 Markdown 拆成标题、段落、表格、代码块等语义块。"""
    lines = markdown.splitlines()
    index = 0
    paragraph: list[str] = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            if paragraph:
                yield "\n".join(paragraph).strip()
                paragraph = []
            index += 1
            continue

        if stripped.startswith("```"):
            if paragraph:
                yield "\n".join(paragraph).strip()
                paragraph = []
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith("```"):
                    index += 1
                    break
                index += 1
            yield "\n".join(block).strip()
            continue

        if _is_markdown_table_start(lines, index):
            if paragraph:
                yield "\n".join(paragraph).strip()
                paragraph = []
            block = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and MARKDOWN_TABLE_LINE.match(lines[index]):
                block.append(lines[index])
                index += 1
            yield "\n".join(block).strip()
            continue

        if stripped.startswith("#"):
            if paragraph:
                yield "\n".join(paragraph).strip()
                paragraph = []
            yield line.strip()
            index += 1
            continue

        paragraph.append(line)
        index += 1

    if paragraph:
        yield "\n".join(paragraph).strip()


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and MARKDOWN_TABLE_LINE.match(lines[index]) is not None
        and MARKDOWN_TABLE_SEPARATOR.match(lines[index + 1]) is not None
    )


def _is_atomic_markdown_block(block: str) -> bool:
    stripped = block.lstrip()
    return stripped.startswith("```") or _is_markdown_table_start(block.splitlines(), 0)


def _split_large_text_block(block: str, chunk_size: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in block.splitlines():
        line_tokens = estimate_token_size(line)
        if current and current_tokens + line_tokens > chunk_size:
            parts.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        parts.append("\n".join(current))
    return parts


def _pick_overlap_blocks(blocks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []
    picked: list[str] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = estimate_token_size(block)
        if picked and total + block_tokens > chunk_overlap:
            break
        picked.insert(0, block)
        total += block_tokens
    return picked
