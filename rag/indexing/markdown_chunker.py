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
    3. H3 原子保护：只对包含特定关键词的 H3/H4 标题生效。
       不超过 H3_ATOMIC_MAX_TOKENS 时整节保留；超过后按常规规则落盘，不丢弃正文。
    4. heading_path 按标题级别入栈。文档 H1 会带入后续章节；一个 chunk 覆盖整节时，
       路径停在覆盖整节的标题，而不是最后一个子标题。
    5. 只有标题、没有正文的碎片并入相邻 chunk，避免占用召回名额。
    """
    blocks = list(split_markdown_blocks(markdown))
    chunks: list[dict[str, str]] = []
    inherited_stack = _leading_heading_stack(blocks)

    # 按 H2 标题分组
    h2_sections = _group_by_h2_sections(blocks)

    # 对每个 H2 章节独立处理。H1 不在分组内部时，从文档开头继承进来。
    for section_blocks in h2_sections:
        section_tokens = sum(estimate_token_size(b) for b in section_blocks)
        section_inherited = _inherited_stack_for_section(section_blocks, inherited_stack)

        if section_tokens <= chunk_size:
            # 策略 1：H2 章节较小，整个作为一个 chunk
            _append_chunk_from_items(chunks, _annotate_blocks(section_blocks, section_inherited))
        elif section_tokens <= chunk_size * 2:
            # 策略 2：中等大小，尝试在 H3 边界切分
            _split_by_h3_boundaries(
                chunks, section_blocks, chunk_size, chunk_overlap, section_inherited
            )
        else:
            # 策略 3：H2 章节很大，应用常规切分 + H3 原子保护
            _split_with_h3_atomic_protection(
                chunks, section_blocks, chunk_size, chunk_overlap, section_inherited
            )

    return _merge_heading_only_chunks(chunks)


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


def _leading_heading_stack(blocks: list[str]) -> list[tuple[int, str]]:
    """取出第一个 H2 之前的标题栈。文档 H1 要带进后面的每个分组。"""
    stack: list[tuple[int, str]] = []
    for block in blocks:
        level = _get_heading_level(block)
        if level <= 0:
            continue
        if level >= 2:
            break
        title = _heading_title(block)
        if title:
            stack = _push_heading(stack, level, title)
    return stack


def _inherited_stack_for_section(
    section_blocks: list[str],
    inherited_stack: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """分组自己已经包含这些标题时，不再从外部重复压入。"""
    if not inherited_stack:
        return []
    root_level, root_title = inherited_stack[0]
    contains_root = any(
        _get_heading_level(block) == root_level and _heading_title(block) == root_title
        for block in section_blocks
    )
    if contains_root:
        return []
    return list(inherited_stack)


def _push_heading(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> list[tuple[int, str]]:
    """按级别维护标题栈。新标题会弹出所有同级和更深的标题，避免平级标题被当成父子。"""
    updated = list(stack)
    while updated and updated[-1][0] >= level:
        updated.pop()
    updated.append((level, title))
    return updated


def _heading_title(block: str) -> str:
    match = re.match(r"^#+\s+(.+?)\s*$", block)
    return match.group(1).strip() if match else ""


def _format_heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def _annotate_blocks(
    blocks: list[str],
    inherited_stack: list[tuple[int, str]],
) -> list[tuple[str, list[tuple[int, str]]]]:
    """给每个块带上它所属的标题栈。栈在遇到标题后立即更新。"""
    stack = list(inherited_stack)
    items: list[tuple[str, list[tuple[int, str]]]] = []
    for block in blocks:
        level = _get_heading_level(block)
        if level > 0:
            title = _heading_title(block)
            if title:
                stack = _push_heading(stack, level, title)
        items.append((block, list(stack)))
    return items


def _append_chunk_from_items(
    chunks: list[dict[str, str]],
    items: list[tuple[str, list[tuple[int, str]]]],
) -> None:
    text = "\n\n".join(block for block, _ in items).strip()
    if not text:
        return
    chunks.append({"text": text, "heading_path": _heading_path_from_items(items)})


def _heading_path_from_items(items: list[tuple[str, list[tuple[int, str]]]]) -> str:
    """一个 chunk 的路径是能覆盖它全部内容的标题。

    标题层层变深、且更深标题前没有正文时，用最深的标题。
    出现平级标题，或父级标题下已经有正文时，停在它们的公共上级，
    这样整节 chunk 不会被标成最后一个子标题。
    """
    if not items:
        return ""
    blocks = [block for block, _ in items]
    stacks = [stack for _, stack in items]
    heading_positions = [index for index, block in enumerate(blocks) if _get_heading_level(block) > 0]
    if not heading_positions:
        return _format_heading_path(stacks[0])

    heading_stacks = [stacks[index] for index in heading_positions]
    last_heading = heading_positions[-1]
    body_before_last = any(
        _get_heading_level(block) == 0 and block.strip()
        for block in blocks[:last_heading]
    )
    if _is_single_heading_chain(heading_stacks) and not body_before_last:
        return _format_heading_path(heading_stacks[-1])
    return _format_heading_path(_common_heading_prefix(stacks))


def _is_single_heading_chain(stacks: list[list[tuple[int, str]]]) -> bool:
    for previous, current in zip(stacks, stacks[1:]):
        if len(current) <= len(previous):
            return False
        if current[: len(previous)] != previous:
            return False
    return True


def _common_heading_prefix(stacks: list[list[tuple[int, str]]]) -> list[tuple[int, str]]:
    if not stacks:
        return []
    prefix = list(stacks[0])
    for stack in stacks[1:]:
        limit = min(len(prefix), len(stack))
        index = 0
        while index < limit and prefix[index] == stack[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _is_heading_only_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return all(_get_heading_level(line) > 0 and _heading_title(line) for line in lines)


def _merge_heading_only_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    """只有标题的 chunk 没有信息量，并入后一个 chunk；末尾则并入前一个。"""
    if len(chunks) <= 1:
        return chunks

    merged: list[dict[str, str]] = []
    pending: list[str] = []
    for chunk in chunks:
        if _is_heading_only_text(chunk["text"]):
            pending.append(chunk["text"])
            continue
        if pending:
            chunk = {
                "text": "\n\n".join(pending + [chunk["text"]]),
                "heading_path": chunk["heading_path"],
            }
            pending = []
        merged.append(chunk)

    if not pending:
        return merged
    tail = "\n\n".join(pending)
    if not merged:
        return [{"text": tail, "heading_path": chunks[-1]["heading_path"]}]
    merged[-1] = {
        "text": merged[-1]["text"] + "\n\n" + tail,
        "heading_path": merged[-1]["heading_path"],
    }
    return merged


def _split_by_h3_boundaries(
    chunks: list[dict[str, str]],
    section_blocks: list[str],
    chunk_size: int,
    chunk_overlap: int,
    inherited_stack: list[tuple[int, str]],
) -> None:
    """在 H3 边界切分中等大小的 H2 章节，优化：避免产生只包含 H2 标题的 chunk"""
    del chunk_overlap  # 这一路按 H3 分组打包，分组边界不再做块级 overlap
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

    heading_stack = list(inherited_stack)
    current_items: list[tuple[str, list[tuple[int, str]]]] = []

    for h3_blocks in h3_groups:
        group_items: list[tuple[str, list[tuple[int, str]]]] = []
        for block in h3_blocks:
            level = _get_heading_level(block)
            if level > 0:
                title = _heading_title(block)
                if title:
                    heading_stack = _push_heading(heading_stack, level, title)
            group_items.append((block, list(heading_stack)))

        h3_tokens = sum(estimate_token_size(block) for block, _ in group_items)
        current_tokens = sum(estimate_token_size(block) for block, _ in current_items)

        # 如果加入当前 H3 会超出 chunk_size，先提交累积的内容
        if current_items and current_tokens + h3_tokens > chunk_size:
            _append_chunk_from_items(chunks, current_items)
            current_items = []

        current_items.extend(group_items)

    if current_items:
        _append_chunk_from_items(chunks, current_items)


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
    inherited_stack: list[tuple[int, str]],
) -> None:
    """对大 H2 章节应用常规切分 + H3 原子保护。"""
    current_items: list[tuple[str, list[tuple[int, str]]]] = []
    heading_stack = list(inherited_stack)

    in_h3_atomic = False
    h3_atomic_items: list[tuple[str, list[tuple[int, str]]]] = []
    h3_atomic_tokens = 0

    for block in section_blocks:
        is_heading = _get_heading_level(block) > 0
        level = _get_heading_level(block)
        block_tokens = estimate_token_size(block)
        if is_heading:
            title = _heading_title(block)
            if title:
                heading_stack = _push_heading(heading_stack, level, title)
        block_stack = list(heading_stack)

        # 检测 H3/H4 原子章节开始
        if is_heading and not in_h3_atomic and level >= 3 and _is_h3_atomic_heading(block):
            # 上级标题很小（通常只有一个 H2）时并入原子章节，避免单独成块
            parent_blocks = [item for item, _ in current_items]
            current_tokens = sum(estimate_token_size(item) for item in parent_blocks)
            should_merge_parent_headings = (
                current_items
                and current_tokens < 50
                and all(_get_heading_level(item) > 0 and _get_heading_level(item) < level for item in parent_blocks)
            )

            if current_items and not should_merge_parent_headings:
                _append_chunk_from_items(chunks, current_items)
                current_items = []

            in_h3_atomic = True
            parent_items = current_items if should_merge_parent_headings else []
            h3_atomic_items = parent_items + [(block, block_stack)]
            h3_atomic_tokens = sum(estimate_token_size(item) for item, _ in parent_items) + block_tokens
            current_items = []
            continue

        # 检测 H3 原子章节结束（遇到同级或更高级标题）
        if in_h3_atomic and is_heading and level <= 3:
            if h3_atomic_tokens <= H3_ATOMIC_MAX_TOKENS:
                _append_chunk_from_items(chunks, h3_atomic_items)
                current_items = []
            else:
                # 超过阈值后按常规规则落盘。必须接住返回的累积列表，不能每处理一块就换成空列表。
                current_items = []
                for atomic_block, atomic_stack in h3_atomic_items:
                    current_items = _append_block_item(
                        chunks,
                        current_items,
                        atomic_block,
                        atomic_stack,
                        chunk_size,
                        chunk_overlap,
                    )

            in_h3_atomic = False
            h3_atomic_items = []
            h3_atomic_tokens = 0

        if in_h3_atomic:
            h3_atomic_items.append((block, block_stack))
            h3_atomic_tokens += block_tokens
            continue

        current_items = _append_block_item(
            chunks,
            current_items,
            block,
            block_stack,
            chunk_size,
            chunk_overlap,
        )

    if in_h3_atomic and h3_atomic_items:
        if h3_atomic_tokens <= H3_ATOMIC_MAX_TOKENS:
            _append_chunk_from_items(chunks, h3_atomic_items)
        else:
            for atomic_block, atomic_stack in h3_atomic_items:
                current_items = _append_block_item(
                    chunks,
                    current_items,
                    atomic_block,
                    atomic_stack,
                    chunk_size,
                    chunk_overlap,
                )

    if current_items:
        _append_chunk_from_items(chunks, current_items)


def _append_block_item(
    chunks: list[dict[str, str]],
    current_items: list[tuple[str, list[tuple[int, str]]]],
    block: str,
    block_stack: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, list[tuple[int, str]]]]:
    """把一个块并入当前 chunk。装满后先落盘，并返回更新后的累积状态。"""
    block_tokens = estimate_token_size(block)
    current_tokens = sum(estimate_token_size(item) for item, _ in current_items)
    if current_items and current_tokens + block_tokens > chunk_size:
        _append_chunk_from_items(chunks, current_items)
        current_items = _pick_overlap_items(current_items, chunk_overlap)

    if block_tokens > chunk_size and not _is_atomic_markdown_block(block):
        if current_items:
            _append_chunk_from_items(chunks, current_items)
            current_items = []
        for part in _split_large_text_block(block, chunk_size):
            part = part.strip()
            if part:
                _append_chunk_from_items(chunks, [(part, block_stack)])
        return current_items

    current_items.append((block, block_stack))
    return current_items


def _pick_overlap_items(
    items: list[tuple[str, list[tuple[int, str]]]],
    chunk_overlap: int,
) -> list[tuple[str, list[tuple[int, str]]]]:
    """选择 overlap，跳过超过 overlap 预算的大表格，优先保留标题和小段落。"""
    if chunk_overlap <= 0:
        return []
    picked: list[tuple[str, list[tuple[int, str]]]] = []
    total = 0
    for item in reversed(items):
        block_tokens = estimate_token_size(item[0])
        if block_tokens > chunk_overlap:
            continue
        if picked and total + block_tokens > chunk_overlap:
            break
        picked.insert(0, item)
        total += block_tokens
    return picked


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
