"""把图片中文说明贴到 Markdown 图片引用后面，供现有切分器一并收下。"""

from __future__ import annotations

import json
import re
from pathlib import Path


CAPTION_PREFIX = "【图片说明｜"
CAPTION_SUFFIX = "】"


def normalize_image_key(path: str) -> str:
    """统一图片引用键，避免斜杠和首尾空白造成对不上。"""
    return path.replace("\\", "/").strip()


def parse_image_line(line: str) -> tuple[str, str, str] | None:
    """解析一行里的 Markdown 图片。路径里允许出现圆括号。

    返回替代文本、图片路径、同一行上已经写好的图片说明。说明和图片写在同一行，
    现有切分器按行切开时不会把二者拆开。
    """
    stripped = line.strip()
    if not stripped.startswith("!["):
        return None
    marker = stripped.find("](")
    if marker < 2:
        return None
    alt = stripped[2:marker]
    rest = stripped[marker + 2 :]
    inline_marker = ") " + CAPTION_PREFIX
    caption = ""
    if inline_marker in rest:
        image_path, _, caption_part = rest.partition(inline_marker)
        caption = _caption_text(CAPTION_PREFIX + caption_part)
    elif rest.endswith(")"):
        image_path = rest[:-1]
    else:
        return None
    image_path = image_path.strip()
    if not image_path or any(char.isspace() for char in image_path):
        return None
    return alt, image_path, caption


def caption_line(image_path: str, description: str) -> str:
    """生成紧跟图片的单行说明，避免被按空行拆成另一个语义块。"""
    name = Path(normalize_image_key(image_path)).name
    text = normalize_caption(description)
    return f"{CAPTION_PREFIX}{name}{CAPTION_SUFFIX}{text}"


def normalize_caption(description: str) -> str:
    return re.sub(r"\s+", " ", description).strip()


def load_image_captions(path: str | Path) -> dict[str, str]:
    """读取图片说明。支持 `{path: text}` 或 `{path: {description: text}}`。"""
    caption_path = Path(path)
    if not caption_path.exists():
        return {}
    payload = json.loads(caption_path.read_text(encoding="utf-8"))
    images = payload.get("images", payload) if isinstance(payload, dict) else {}
    if not isinstance(images, dict):
        raise ValueError(f"图片说明格式不正确: {caption_path}")
    captions: dict[str, str] = {}
    for key, value in images.items():
        description = _description_text(value)
        if description:
            captions[normalize_image_key(str(key))] = description
    return captions


def inject_image_captions(markdown: str, captions: dict[str, str]) -> tuple[str, list[str]]:
    """把说明写到图片同一行末尾。返回新 Markdown，以及没有说明的图片键。"""
    lines = markdown.splitlines()
    output: list[str] = []
    missing: list[str] = []
    for line in lines:
        parsed = parse_image_line(line)
        if parsed is None:
            output.append(line)
            continue
        _, image_path, existing = parsed
        image_key = normalize_image_key(image_path)
        if existing:
            output.append(line)
            continue
        description = captions.get(image_key, "")
        if description:
            output.append(f"{line.rstrip()} {caption_line(image_path, description)}")
        else:
            output.append(line)
            missing.append(image_key)
    text = "\n".join(output).strip()
    return (text + "\n") if text else "", missing


def extract_chunk_images(text: str) -> list[dict[str, str]]:
    """从 chunk 正文提取图片路径和紧随的中文说明。"""
    images: list[dict[str, str]] = []
    for line in text.splitlines():
        parsed = parse_image_line(line)
        if parsed is None:
            continue
        _, image_path, description = parsed
        image_key = normalize_image_key(image_path)
        images.append(
            {
                "path": image_key,
                "file_name": Path(image_key).name,
                "description": description,
            }
        )
    return images


def iter_markdown_image_keys(markdown: str) -> list[str]:
    keys: list[str] = []
    for line in markdown.splitlines():
        parsed = parse_image_line(line)
        if parsed is not None:
            keys.append(normalize_image_key(parsed[1]))
    return keys


def _description_text(value: object) -> str:
    if isinstance(value, str):
        return normalize_caption(value)
    if isinstance(value, dict):
        return normalize_caption(str(value.get("description") or ""))
    return ""


def _caption_text(line: str) -> str:
    stripped = line.strip()
    marker = stripped.find(CAPTION_SUFFIX)
    if marker < 0:
        return ""
    return stripped[marker + len(CAPTION_SUFFIX) :].strip()
