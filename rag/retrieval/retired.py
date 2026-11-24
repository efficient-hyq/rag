"""识别并排除明确标记为已废弃或已过时的文档和章节。"""

from __future__ import annotations

import re
from typing import Any

# 只认文档路径、文件名和标题上的明确状态，避免正文里偶然出现“废弃”就被整篇丢掉。
_STATUS_MARKERS = ("已废弃", "已过时", "已经过时", "（废弃）", "(废弃)")
_PATH_FIELDS = (
    "archive_path",
    "markdown_relative_path",
    "cleaned_markdown_relative_path",
    "file_name",
    "doc_id",
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_STRUCK_RE = re.compile(r"~~.+?~~")


def is_retired_document(metadata: dict[str, Any]) -> bool:
    """路径或文件名标明整篇文档已废弃或已过时。"""
    blob = "\n".join(str(metadata.get(key) or "") for key in _PATH_FIELDS)
    return any(marker in blob for marker in _STATUS_MARKERS)


def is_retired_heading(title: str) -> bool:
    """标题本身标明章节已废弃、已过时，或整段标题被删除线划掉。"""
    text = str(title or "").strip()
    if not text:
        return False
    if any(marker in text for marker in _STATUS_MARKERS):
        return True
    if "~~" not in text:
        return False
    if "废弃" in text or "过时" in text:
        return True
    rest = _STRUCK_RE.sub("", text)
    rest = re.sub(r"[\s:：()（）\-—]+", "", rest)
    return not rest


def usable_text(metadata: dict[str, Any], text: str | None = None) -> str | None:
    """返回可以交给模型的正文。整篇废弃，或去掉过时章节后没有剩余内容时返回 None。"""
    if is_retired_document(metadata):
        return None
    body = str(metadata.get("text") or "") if text is None else str(text)
    return strip_retired_sections(body, str(metadata.get("heading_path") or ""))


def strip_retired_sections(text: str, heading_path: str = "") -> str | None:
    """按标题去掉已废弃或已过时的章节，保留同一片段里的现行内容。"""
    raw = str(text or "")
    if not raw.strip():
        return None
    leaf = _heading_leaf(heading_path)
    retired = is_retired_heading(leaf)
    kept: list[str] = []
    buffer: list[str] = []
    for line in raw.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue
        _flush(retired, buffer, kept)
        buffer = [line]
        retired = is_retired_heading(match.group(2))
    _flush(retired, buffer, kept)
    cleaned = "\n".join(kept).strip()
    return cleaned or None


def exclude_retired_candidates(candidates: list[Any]) -> tuple[list[Any], int]:
    """丢掉不可用片段，并把混合片段里的废弃章节从正文中去掉。"""
    kept: list[Any] = []
    dropped = 0
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        cleaned = usable_text(metadata, getattr(candidate, "text", ""))
        if cleaned is None:
            dropped += 1
            continue
        if cleaned != candidate.text:
            candidate.text = cleaned
            candidate.metadata["text"] = cleaned
        kept.append(candidate)
    return kept, dropped


def _heading_leaf(heading_path: str) -> str:
    parts = [part.strip() for part in str(heading_path or "").split(">") if part.strip()]
    return parts[-1] if parts else ""


def _flush(retired: bool, buffer: list[str], kept: list[str]) -> None:
    if retired or not any(line.strip() for line in buffer):
        buffer.clear()
        return
    if kept and kept[-1] != "":
        kept.append("")
    kept.extend(buffer)
    buffer.clear()
