from __future__ import annotations

"""
Word/Confluence 导出文档预处理脚本。

@author heyuqin
@date 2026/05/07
"""

import argparse
import hashlib
import html
import json
import logging
import quopri
import shutil
import subprocess
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Comment

from rag.shared.logging_utils import configure_console_logging, log_phase


SUPPORTED_WORD_SUFFIXES = {".doc", ".docx"}
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
IMAGE_ASSET_MANIFEST_FILENAME = "image_manifest.json"
NOISE_TAGS = {"script", "style", "meta", "link", "noscript", "svg"}
NOISE_CLASS_OR_ID_PATTERNS = (
    "page-metadata",
    "page-metadata-banner",
    "content-byline",
    "toc",
    "toc-indentation",
    "TOCOutline",
    "toc-macro",
    "confluenceTableSmall",
    "print-only",
    "hidden",
)


@dataclass(frozen=True)
class PreprocessConfig:
    """文档预处理配置。"""

    input_dir: Path
    html_dir: Path
    markdown_dir: Path
    image_dir: Path = Path("./storage/cleaned_assets")
    recursive: bool = True
    soffice_path: str | None = None


@dataclass(frozen=True)
class PreprocessResult:
    """文档预处理结果。"""

    source_count: int
    markdown_count: int
    markdown_dir: Path
    markdown_files: list[Path]
    source_manifest_path: Path


@dataclass(frozen=True)
class ExtractedAsset:
    reference: str
    content_type: str
    data: bytes
    file_name: str | None = None
    content_id: str | None = None


@dataclass(frozen=True)
class ExtractedMhtmlPackage:
    html_text: str
    assets: list[ExtractedAsset]


def preprocess_documents(config: PreprocessConfig) -> PreprocessResult:
    """执行 doc/docx -> HTML -> Markdown 的文档清洗链路。"""
    logger = logging.getLogger("rag.preprocess")
    input_root = config.input_dir
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    soffice = resolve_soffice(config.soffice_path)
    with log_phase(logger, "扫描原始文档", input_dir=str(input_root)):
        word_files = list(iter_word_files(input_root, recursive=config.recursive))
    if not word_files:
        raise FileNotFoundError(f"未发现 doc/docx 文件: {input_root}")
    logger.info("扫描完成 | source_count=%s", len(word_files))

    config.html_dir.mkdir(parents=True, exist_ok=True)
    config.markdown_dir.mkdir(parents=True, exist_ok=True)
    config.image_dir.mkdir(parents=True, exist_ok=True)

    markdown_files: list[Path] = []
    source_records: list[dict[str, str]] = []
    for source_path in word_files:
        relative_path = source_path.relative_to(input_root)
        html_work_dir = config.html_dir / relative_path.parent / _stable_work_dir_name(source_path)
        html_work_dir.mkdir(parents=True, exist_ok=True)
        image_work_dir = config.image_dir / relative_path.parent / _stable_work_dir_name(source_path)
        image_work_dir.mkdir(parents=True, exist_ok=True)
        asset_manifest_path: Path | None = None
        mhtml_package = extract_mhtml_package_from_file(source_path)
        if mhtml_package is not None:
            asset_mapping = write_extracted_assets(
                mhtml_package.assets,
                image_work_dir,
                path_prefix=_safe_relative_path(image_work_dir, config.image_dir.parent),
            )
            raw_html = rewrite_html_image_sources(mhtml_package.html_text, asset_mapping)
            asset_manifest_path = write_asset_manifest(image_work_dir, asset_mapping)
            html_path = write_extracted_html(source_path, relative_path, config.html_dir, raw_html)
        else:
            html_path = convert_word_to_html(source_path, relative_path, config.html_dir, soffice)
            raw_html = extract_embedded_html(html_path.read_text(encoding="utf-8", errors="ignore"))

        cleaned_html = clean_html(raw_html)
        markdown = html_to_markdown(cleaned_html)
        markdown = normalize_markdown(markdown)

        markdown_path = config.markdown_dir / relative_path.with_suffix(".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        markdown_files.append(markdown_path)
        source_records.append(
            build_source_record(
                source_path=source_path,
                html_path=html_path,
                markdown_path=markdown_path,
                input_root=input_root,
                markdown_root=config.markdown_dir,
                asset_manifest_path=asset_manifest_path,
            )
        )
        logger.info("已清洗 | source=%s | markdown=%s", source_path, markdown_path)

    source_manifest_path = write_source_manifest(config.markdown_dir, source_records)
    logger.info("预处理完成 | source_count=%s | markdown_count=%s", len(word_files), len(markdown_files))
    return PreprocessResult(
        source_count=len(word_files),
        markdown_count=len(markdown_files),
        markdown_dir=config.markdown_dir,
        markdown_files=markdown_files,
        source_manifest_path=source_manifest_path,
    )


def resolve_soffice(explicit_path: str | None = None) -> str:
    """定位 LibreOffice/OpenOffice 的 soffice 命令。"""
    candidates = [
        explicit_path,
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "未找到 LibreOffice soffice。请安装 LibreOffice，或通过 --soffice-path 指定 soffice.exe"
    )


def iter_word_files(input_dir: Path, recursive: bool = True) -> Iterable[Path]:
    """遍历待清洗的 Word 文档。"""
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_WORD_SUFFIXES:
            yield path


def convert_word_to_html(source_path: Path, relative_path: Path, html_root: Path, soffice: str) -> Path:
    """使用 LibreOffice 将 doc/docx 转为 HTML。"""
    work_dir = html_root / relative_path.parent / _stable_work_dir_name(source_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        soffice,
        "--headless",
        "--convert-to",
        "html",
        "--outdir",
        str(work_dir),
        str(source_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Word 转 HTML 失败: {source_path}\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    html_candidates = sorted(work_dir.glob("*.html")) + sorted(work_dir.glob("*.htm"))
    if not html_candidates:
        raise FileNotFoundError(f"LibreOffice 未生成 HTML 文件: {source_path}")
    return html_candidates[0]


def extract_mhtml_package_from_file(source_path: Path) -> ExtractedMhtmlPackage | None:
    """从 Confluence MHTML 风格的 doc 中提取 HTML 正文和图片附件。"""
    try:
        message = BytesParser(policy=policy.default).parsebytes(source_path.read_bytes())
    except Exception:
        return None
    if not message.is_multipart():
        return None

    html_parts: list[str] = []
    assets: list[ExtractedAsset] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        content_type = part.get_content_type().lower()
        if content_type == "text/html":
            content = _decode_text_part(part)
            if content:
                html_parts.append(str(content))
            continue
        if content_type.startswith("image/"):
            asset = _extract_asset(part)
            if asset is not None:
                assets.append(asset)

    if not html_parts:
        return None
    return ExtractedMhtmlPackage(html_text=max(html_parts, key=len), assets=assets)


def write_extracted_html(source_path: Path, relative_path: Path, html_root: Path, html_text: str) -> Path:
    """把直接解出的 HTML 写入中间目录，便于排查清洗质量。"""
    work_dir = html_root / relative_path.parent / _stable_work_dir_name(source_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / relative_path.with_suffix(".html").name
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def write_extracted_assets(
    assets: list[ExtractedAsset],
    output_dir: Path,
    path_prefix: str | None = None,
) -> dict[str, str]:
    """把提取出的图片附件写入目录，并返回原始引用到文件路径的映射。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    image_index = 0
    for asset in assets:
        if not asset.data:
            continue
        image_index += 1
        suffix = _asset_suffix(asset.content_type, asset.file_name)
        file_name = f"img_{image_index:04d}{suffix}"
        asset_path = output_dir / file_name
        asset_path.write_bytes(asset.data)
        relative_path = f"{path_prefix.rstrip('/')}/{file_name}" if path_prefix else asset_path.as_posix()
        for reference in _asset_references(asset):
            mapping[reference] = relative_path
    return mapping


def write_asset_manifest(output_dir: Path, asset_mapping: dict[str, str]) -> Path:
    manifest_path = output_dir / IMAGE_ASSET_MANIFEST_FILENAME
    payload = {
        "assets": [
            {"reference": reference, "path": path}
            for reference, path in sorted(asset_mapping.items())
        ]
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def rewrite_html_image_sources(html_text: str, asset_mapping: dict[str, str]) -> str:
    """将 HTML 中的 attach/cid 图片引用重写为真实落盘路径。"""
    if not asset_mapping:
        return html_text

    normalized_mapping = {
        _normalize_asset_reference(reference): path
        for reference, path in asset_mapping.items()
        if _normalize_asset_reference(reference)
    }
    soup = BeautifulSoup(html_text, "lxml")
    image_index = 0
    for image in soup.find_all("img"):
        reference = str(image.get("src") or image.get("data-image-src") or "").strip()
        if not reference:
            continue
        normalized_reference = _normalize_asset_reference(reference)
        resolved = normalized_mapping.get(normalized_reference)
        if resolved is None and "/" not in normalized_reference:
            resolved = normalized_mapping.get(f"c:/{normalized_reference}")
        if resolved is None:
            continue
        image_index += 1
        image["src"] = resolved
        image["data-image-src"] = resolved
        image["data-image-id"] = f"img_{image_index:04d}"
    body = soup.body or soup
    return str(body)


def clean_html(raw_html: str) -> str:
    """清洗 Word/Confluence HTML 中对检索无价值的噪声。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        if tag.parent is None or tag.attrs is None:
            continue
        if _is_hidden(tag) or _looks_like_export_noise(tag):
            tag.decompose()
            continue
        if tag.name in {"font", "span"} and not tag.get_text(strip=True):
            tag.decompose()

    for table_index, table in enumerate(soup.find_all("table"), start=1):
        if not table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "caption"]):
            marker = soup.new_tag("p")
            marker.string = f"表格 {table_index}"
            table.insert_before(marker)

    body = soup.body or soup
    return str(body)


def extract_embedded_html(converted_html: str) -> str:
    """从 LibreOffice 包裹的 Confluence MHTML 文本中解出真实 HTML。"""
    soup = BeautifulSoup(converted_html, "lxml")
    pre_text = "\n".join(pre.get_text() for pre in soup.find_all("pre"))
    if not _looks_like_mhtml_export(pre_text):
        return converted_html

    embedded_html = decode_mhtml_html_part(pre_text)
    return embedded_html or converted_html


def decode_mhtml_html_part(text: str) -> str:
    """解析 Confluence 导出的 MHTML 文本，解码 quoted-printable HTML 正文。"""
    normalized = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    html_lines: list[str] = []
    in_html_part = False
    reading_body = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("content-type: text/html"):
            in_html_part = True
            reading_body = False
            html_lines = []
            continue
        if in_html_part and not reading_body:
            if not stripped:
                reading_body = True
            continue
        if in_html_part and reading_body:
            if stripped.startswith("------=_") or lower.startswith("content-type:"):
                break
            html_lines.append(line)

    if not html_lines:
        return ""

    decoded = quopri.decodestring("\n".join(html_lines).encode("utf-8")).decode(
        "utf-8",
        errors="ignore",
    )
    start = decoded.lower().find("<html")
    if start >= 0:
        decoded = decoded[start:]
    end = decoded.lower().rfind("</html>")
    if end >= 0:
        decoded = decoded[: end + len("</html>")]
    return decoded


def html_to_markdown(raw_html: str) -> str:
    """将清洗后的 HTML 转为 GitHub Flavored Markdown。"""
    try:
        from markdownify import markdownify
    except ImportError as exc:
        raise RuntimeError("缺少 markdownify 依赖，请先执行: pip install -r requirements.txt") from exc

    return markdownify(
        raw_html,
        heading_style="ATX",
        bullets="-",
        strip=["span", "font"],
    )


def normalize_markdown(markdown: str) -> str:
    """规整 Markdown 空白字符，减少后续分块噪声。"""
    text = (
        markdown.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    blank_count = 0
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            normalized.append(line)
            blank_count = 0
            continue
        if in_fence:
            normalized.append(line)
            continue
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                normalized.append("")
            continue
        blank_count = 0
        normalized.append(line)
    return "\n".join(normalized).strip() + "\n"


def build_source_record(
    source_path: Path,
    html_path: Path,
    markdown_path: Path,
    input_root: Path,
    markdown_root: Path,
    asset_manifest_path: Path | None = None,
) -> dict[str, str]:
    """构建清洗前后文件路径映射，保证后续 chunk 可追溯原始文档。"""
    record = {
        "source_path": str(source_path),
        "source_absolute_path": str(source_path.resolve()),
        "source_relative_path": _safe_relative_path(source_path, input_root),
        "source_file_name": source_path.name,
        "source_file_type": source_path.suffix.lower().lstrip("."),
        "converted_html_path": str(html_path),
        "converted_html_absolute_path": str(html_path.resolve()),
        "markdown_path": str(markdown_path),
        "markdown_absolute_path": str(markdown_path.resolve()),
        "markdown_relative_path": _safe_relative_path(markdown_path, markdown_root),
    }
    if asset_manifest_path is not None:
        record["image_asset_manifest_path"] = str(asset_manifest_path)
        record["image_asset_manifest_absolute_path"] = str(asset_manifest_path.resolve())
    return record


def write_source_manifest(markdown_root: Path, records: list[dict[str, str]]) -> Path:
    """写入 Markdown 与原始文件路径映射清单。"""
    manifest_path = markdown_root / SOURCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps({"documents": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 doc/docx 并生成 Markdown 文档")
    parser.add_argument("--input-dir", default="C:/Users/heyuqin/Desktop/RAG_DATA", help="原始 doc/docx 文档目录")
    parser.add_argument("--html-dir", default="./storage/converted_html", help="中间 HTML 输出目录")
    parser.add_argument("--markdown-dir", default="./storage/cleaned_markdown", help="清洗后 Markdown 输出目录")
    parser.add_argument("--image-dir", default="./storage/cleaned_assets", help="提取图片资产输出目录")
    parser.add_argument("--soffice-path", default=None, help="LibreOffice soffice.exe 路径")
    parser.add_argument("--no-recursive", action="store_true", help="仅处理输入目录一级文件")
    return parser.parse_args()


def main() -> None:
    configure_console_logging()
    args = parse_args()
    result = preprocess_documents(
        PreprocessConfig(
            input_dir=Path(args.input_dir),
            html_dir=Path(args.html_dir),
            markdown_dir=Path(args.markdown_dir),
            image_dir=Path(args.image_dir),
            recursive=not args.no_recursive,
            soffice_path=args.soffice_path,
        )
    )
    print(
        f"预处理完成: source={result.source_count}, markdown={result.markdown_count}, "
        f"markdown_dir={result.markdown_dir}, manifest={result.source_manifest_path}",
        flush=True,
    )


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_work_dir_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{digest}"


def _decode_text_part(part: Message) -> str:
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        content = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
    return str(content or "")


def _extract_asset(part: Message) -> ExtractedAsset | None:
    payload = part.get_payload(decode=True) or b""
    if not payload:
        return None
    return ExtractedAsset(
        reference=_pick_asset_reference(part),
        content_type=part.get_content_type().lower(),
        data=payload,
        file_name=part.get_filename(),
        content_id=_normalize_asset_reference(str(part.get("Content-ID") or "")),
    )


def _pick_asset_reference(part: Message) -> str:
    for key in ("Content-Location", "X-Attachment-Id", "Content-ID"):
        value = part.get(key)
        if value:
            return _normalize_asset_reference(str(value))
    if part.get_filename():
        return _normalize_asset_reference(str(part.get_filename()))
    digest = hashlib.sha256((part.get_payload(decode=True) or b"")).hexdigest()[:16]
    return f"asset_{digest}"


def _asset_references(asset: ExtractedAsset) -> list[str]:
    values = [asset.reference]
    if asset.content_id:
        values.append(asset.content_id)
    if asset.file_name:
        values.append(_normalize_asset_reference(asset.file_name))
    values.extend(_alternate_asset_references(values))
    normalized = [_normalize_asset_reference(value) for value in values if value]
    return list(dict.fromkeys(normalized))


def _normalize_asset_reference(value: str) -> str:
    normalized = value.strip().strip("<>").replace("cid:", "").strip()
    if normalized.lower().startswith("file:///"):
        normalized = normalized[8:]
    normalized = normalized.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        normalized = normalized[0].lower() + normalized[1:]
    return normalized


def _asset_suffix(content_type: str, file_name: str | None) -> str:
    if file_name:
        suffix = Path(file_name).suffix
        if suffix:
            return suffix.lower()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    return mapping.get(content_type.lower(), ".bin")


def _alternate_asset_references(values: list[str]) -> list[str]:
    alternates: list[str] = []
    for value in values:
        normalized = _normalize_asset_reference(value)
        if not normalized:
            continue
        lower = normalized.lower()
        if "/attach_" in lower:
            alternates.append(normalized.split("/")[-1])
        elif normalized.startswith("C:/") or normalized.startswith("c:/"):
            alternates.append(normalized.split("/")[-1])
    return alternates


def _looks_like_mhtml_export(text: str) -> bool:
    lower = text.lower()
    return (
        "message-id:" in lower
        and "content-transfer-encoding: quoted-printable" in lower
        and "content-type: text/html" in lower
    )


def _is_hidden(tag) -> bool:
    style = str(tag.get("style") or "").replace(" ", "").lower()
    return (
        tag.get("hidden") is not None
        or str(tag.get("aria-hidden") or "").lower() == "true"
        or "display:none" in style
        or "visibility:hidden" in style
    )


def _looks_like_export_noise(tag) -> bool:
    values: list[str] = []
    for attr_name in ("class", "id", "data-macro-name"):
        attr_value = tag.get(attr_name)
        if isinstance(attr_value, list):
            values.extend(str(item) for item in attr_value)
        elif attr_value:
            values.append(str(attr_value))
    joined = " ".join(values).lower()
    return any(pattern.lower() in joined for pattern in NOISE_CLASS_OR_ID_PATTERNS)


if __name__ == "__main__":
    main()
