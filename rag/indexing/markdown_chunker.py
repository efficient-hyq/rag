from __future__ import annotations

import re
from typing import Iterable

from llama_index.core.schema import Document, TextNode

from rag.retrieval.tokenization import estimate_token_size
from rag.shared.checkpoints import stable_chunk_id


MARKDOWN_TABLE_LINE = re.compile(r"^\s*\|.+\|\s*$")
MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

# H3 原子章节关键词：只对三级标题及以下应用原子保护
# 这些章节通常包含紧密相关的结构化内容（表格、代码、枚举等）
H3_ATOMIC_KEYWORDS = [
    "接口",  # 接口类标题（如"生成预支付订单接口"）
    "加密",
    "解密",
    "签名",
    "验签",
    "错误码",
    "状态码",
    "返回码",
    "字段说明",
    "数据结构",
    "枚举",
    "常量",
    "请求参数",
    "响应参数",
    "请求体",
    "响应体",
]

# H3 原子章节的最大 token 限制（超过则强制切分）
# 对于接口文档，单个接口的表格可能较大，适当提高限制
H3_ATOMIC_MAX_TOKENS = 1000


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
        chunks = split_markdown_chunks(document.get_content(), chunk_size, chunk_overlap)
        for chunk_index, item in enumerate(chunks):
            chunk_text = item["text"]
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["token_size"] = estimate_token_size(chunk_text)
            chunk_metadata["heading_path"] = item["heading_path"]
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
    return [item["text"] for item in split_markdown_chunks(markdown, chunk_size, chunk_overlap)]


def split_markdown_chunks(markdown: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, str]]:
    """按 H2 边界切分，H3 原子章节保护，提升 API 文档检索精度。

    切分策略（方案四：混合策略）：
    1. H2 标题作为强边界：每个 H2 章节独立处理（API 接口通常是 H2）
    2. H2 章节内部：
       - 小于 chunk_size：整个 H2 章节作为一个 chunk
       - 大于 chunk_size 但小于 2*chunk_size：尝试在 H3 边界切分
       - 远大于 2*chunk_size：按常规逻辑切分，H3 原子章节保护
    3. H3 原子保护：只对包含特定关键词的 H3/H4 标题生效（<800 tokens）
    """
    blocks = list(split_markdown_blocks(markdown))
    chunks: list[dict[str, str]] = []

    # 按 H2 标题分组
    h2_sections = _group_by_h2_sections(blocks)

    # 对每个 H2 章节独立处理
    for section_blocks in h2_sections:
        section_tokens = sum(estimate_token_size(b) for b in section_blocks)
        section_heading_path = _extract_heading_path(section_blocks)

        if section_tokens <= chunk_size:
            # 策略 1：H2 章节较小，整个作为一个 chunk
            _append_chunk(chunks, section_blocks, section_heading_path)
        elif section_tokens <= chunk_size * 2:
            # 策略 2：中等大小，尝试在 H3 边界切分
            _split_by_h3_boundaries(chunks, section_blocks, chunk_size, chunk_overlap)
        else:
            # 策略 3：H2 章节很大，应用常规切分 + H3 原子保护
            _split_with_h3_atomic_protection(chunks, section_blocks, chunk_size, chunk_overlap)

    return chunks


def _group_by_h2_sections(blocks: list[str]) -> list[list[str]]:
    """将 blocks 按 H2 标题分组"""
    sections: list[list[str]] = []
    current_section: list[str] = []

    for block in blocks:
        if re.match(r"^##\s+", block):  # H2 标题
            if current_section:
                sections.append(current_section)
            current_section = [block]
        else:
            current_section.append(block)

    if current_section:
        sections.append(current_section)

    return sections


def _extract_heading_path(blocks: list[str]) -> str:
    """从 blocks 中提取标题路径"""
    heading_stack: list[str] = []
    for block in blocks:
        match = re.match(r"^(#+)\s+(.+?)\s*$", block)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
    return " > ".join(heading_stack)


def _append_chunk(chunks: list[dict[str, str]], blocks: list[str], path: str) -> None:
    """添加一个 chunk"""
    text = "\n\n".join(blocks).strip()
    if text:
        chunks.append({"text": text, "heading_path": path})


def _split_by_h3_boundaries(
    chunks: list[dict[str, str]],
    section_blocks: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """在 H3 边界切分中等大小的 H2 章节，优化：避免产生只包含 H2 标题的 chunk"""
    h3_groups = _group_by_h3_within_section(section_blocks)

    # 优化：如果第一个组只包含 H2 标题（很小），将其合并到第二个组
    if len(h3_groups) >= 2:
        first_group = h3_groups[0]
        first_group_tokens = sum(estimate_token_size(b) for b in first_group)
        # 检查是否只包含 H2（或其他上级标题），且很小
        if first_group_tokens < 50 and all(_get_heading_level(b) <= 2 for b in first_group):
            # 合并到第二个组
            h3_groups[1] = first_group + h3_groups[1]
            h3_groups = h3_groups[1:]  # 移除第一个组

    current_blocks: list[str] = []
    current_tokens = 0

    for h3_blocks in h3_groups:
        h3_tokens = sum(estimate_token_size(b) for b in h3_blocks)

        # 如果加入当前 H3 会超出 chunk_size，先提交累积的内容
        if current_blocks and current_tokens + h3_tokens > chunk_size:
            _append_chunk(chunks, current_blocks, _extract_heading_path(current_blocks))
            current_blocks = []
            current_tokens = 0

        current_blocks.extend(h3_blocks)
        current_tokens += h3_tokens

    if current_blocks:
        _append_chunk(chunks, current_blocks, _extract_heading_path(current_blocks))


def _group_by_h3_within_section(section_blocks: list[str]) -> list[list[str]]:
    """在 H2 章节内按 H3 分组"""
    groups: list[list[str]] = []
    current_group: list[str] = []

    for block in section_blocks:
        level = _get_heading_level(block)
        if level == 3:  # H3 标题
            if current_group:
                groups.append(current_group)
            current_group = [block]
        else:
            current_group.append(block)

    if current_group:
        groups.append(current_group)

    return groups


def _split_with_h3_atomic_protection(
    chunks: list[dict[str, str]],
    section_blocks: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """对大 H2 章节应用常规切分 + H3 原子保护"""
    current_blocks: list[str] = []
    current_tokens = 0
    current_heading_path = ""
    heading_stack: list[str] = []

    # H3 原子章节状态
    in_h3_atomic = False
    h3_atomic_blocks: list[str] = []
    h3_atomic_tokens = 0
    h3_atomic_path = ""

    for block in section_blocks:
        is_heading = re.match(r"^#+\s+", block) is not None
        level = _get_heading_level(block)
        block_tokens = estimate_token_size(block)

        # 更新标题栈
        if is_heading:
            match = re.match(r"^#+\s+(.+?)\s*$", block)
            if match:
                title = match.group(1).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)

        block_heading_path = " > ".join(heading_stack)

        # 检测 H3/H4 原子章节开始
        if is_heading and not in_h3_atomic and level >= 3 and _is_h3_atomic_heading(block):
            # 优化：如果 current_blocks 只包含上级标题且很小（如单个 H2），将其并入 H3 原子章节
            # 避免产生信息量极少的 chunk（如只有一个 H2 标题）
            should_merge_parent_headings = (
                current_blocks
                and current_tokens < 50  # 很小，通常只是标题
                and all(_get_heading_level(b) > 0 and _get_heading_level(b) < level for b in current_blocks)
            )

            if current_blocks and not should_merge_parent_headings:
                # 提交之前的内容（如果不是要合并的上级标题）
                _append_chunk(chunks, current_blocks, current_heading_path)
                current_blocks = []
                current_tokens = 0

            # 开始 H3 原子章节
            in_h3_atomic = True
            h3_atomic_blocks = current_blocks + [block]  # 可能包含上级标题
            h3_atomic_tokens = current_tokens + block_tokens
            h3_atomic_path = block_heading_path
            current_blocks = []
            current_tokens = 0
            continue

        # 检测 H3 原子章节结束（遇到同级或更高级标题）
        if in_h3_atomic and is_heading and level <= 3:
            # 提交 H3 原子章节
            if h3_atomic_tokens <= H3_ATOMIC_MAX_TOKENS:
                _append_chunk(chunks, h3_atomic_blocks, h3_atomic_path)
            else:
                # 超过阈值，降级为常规切分
                for atomic_block in h3_atomic_blocks:
                    _regular_chunking_append(
                        chunks, current_blocks, current_tokens, current_heading_path,
                        atomic_block, h3_atomic_path, chunk_size, chunk_overlap
                    )
                    current_blocks, current_tokens, current_heading_path = [], 0, h3_atomic_path

            # 退出 H3 原子模式
            in_h3_atomic = False
            h3_atomic_blocks = []
            h3_atomic_tokens = 0

        # 在 H3 原子章节内，持续累积
        if in_h3_atomic:
            h3_atomic_blocks.append(block)
            h3_atomic_tokens += block_tokens
            continue

        # 常规切分逻辑
        if current_blocks and current_tokens + block_tokens > chunk_size:
            _append_chunk(chunks, current_blocks, current_heading_path)
            overlap_blocks = _pick_overlap_blocks(current_blocks, chunk_overlap)
            current_blocks = overlap_blocks[:]
            current_tokens = sum(estimate_token_size(b) for b in current_blocks)
            if not current_blocks:
                current_heading_path = block_heading_path

        if not current_blocks or is_heading:
            current_heading_path = block_heading_path

        if block_tokens > chunk_size and not _is_atomic_markdown_block(block):
            for part in _split_large_text_block(block, chunk_size):
                if current_blocks:
                    _append_chunk(chunks, current_blocks, current_heading_path)
                    current_blocks = []
                    current_tokens = 0
                _append_chunk(chunks, [part.strip()], current_heading_path)
            continue

        current_blocks.append(block)
        current_tokens += block_tokens

    # 处理剩余的 H3 原子章节
    if in_h3_atomic and h3_atomic_blocks:
        if h3_atomic_tokens <= H3_ATOMIC_MAX_TOKENS:
            _append_chunk(chunks, h3_atomic_blocks, h3_atomic_path)
        else:
            for atomic_block in h3_atomic_blocks:
                _regular_chunking_append(
                    chunks, current_blocks, current_tokens, current_heading_path,
                    atomic_block, h3_atomic_path, chunk_size, chunk_overlap
                )
                current_blocks, current_tokens = [], 0

    # 处理剩余的常规块
    if current_blocks:
        _append_chunk(chunks, current_blocks, current_heading_path)


def _regular_chunking_append(
    chunks: list[dict[str, str]],
    current_blocks: list[str],
    current_tokens: int,
    current_heading_path: str,
    block: str,
    block_heading_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """常规切分时添加 block"""
    block_tokens = estimate_token_size(block)
    if current_blocks and current_tokens + block_tokens > chunk_size:
        _append_chunk(chunks, current_blocks, current_heading_path)
        current_blocks.clear()
    current_blocks.append(block)


def _is_h3_atomic_heading(block: str) -> bool:
    """判断 H3/H4 标题是否需要原子保护"""
    match = re.match(r"^#+\s+(.+?)\s*$", block)
    if not match:
        return False
    heading_text = match.group(1).strip().lower()
    return any(keyword in heading_text for keyword in H3_ATOMIC_KEYWORDS)


def _get_heading_level(block: str) -> int:
    """获取标题级别，非标题返回 0"""
    match = re.match(r"^(#+)\s+", block)
    return len(match.group(1)) if match else 0


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
    """选择 overlap 块，但跳过大表格以避免重复。

    策略：
    1. 如果 block 是大表格（超过 overlap 限制），跳过
    2. 优先选择标题和小段落作为 overlap
    """
    if chunk_overlap <= 0:
        return []
    picked: list[str] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = estimate_token_size(block)

        # 跳过超大块（主要是大表格），避免重复
        if block_tokens > chunk_overlap:
            continue

        if picked and total + block_tokens > chunk_overlap:
            break
        picked.insert(0, block)
        total += block_tokens
    return picked
