from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llama_index.core import Document, SimpleDirectoryReader


SUPPORTED_SUFFIXES = {".md"}
SOURCE_MANIFEST_FILENAME = "source_manifest.json"


@dataclass(frozen=True)
class MarkdownDocumentDiff:
    added: set[str]
    changed: set[str]
    deleted: set[str]
    unchanged: set[str]


def load_documents(input_dir: str | Path, recursive: bool = True) -> list[Document]:
    """加载清洗后的 Markdown 文档，并补齐设计要求中的来源元数据。"""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"文档目录不存在: {root}")

    source_manifest = load_source_manifest(root)
    reader = SimpleDirectoryReader(
        input_dir=str(root),
        recursive=recursive,
        filename_as_id=True,
        required_exts=sorted(SUPPORTED_SUFFIXES),
    )
    documents = reader.load_data()
    return [
        _normalize_document_metadata(document, root, source_manifest)
        for document in documents
    ]


def load_documents_from_files(files: list[Path], root: str | Path) -> list[Document]:
    """只加载指定 Markdown 文件，用于文档级增量重建。"""
    if not files:
        return []

    docs_root = Path(root)
    source_manifest = load_source_manifest(docs_root)
    reader = SimpleDirectoryReader(
        input_files=[str(path) for path in files],
        filename_as_id=True,
        required_exts=sorted(SUPPORTED_SUFFIXES),
    )
    documents = reader.load_data()
    return [
        _normalize_document_metadata(document, docs_root, source_manifest)
        for document in documents
    ]


def normalize_doc_key(path: Path, root: str | Path) -> str:
    return path.resolve().relative_to(Path(root).resolve()).as_posix().lower()


def compute_document_content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_current_markdown_state(root: str | Path) -> dict[str, str]:
    return {
        doc_key: compute_document_content_hash(path)
        for doc_key, path in collect_current_markdown_files(root).items()
    }


def collect_current_markdown_files(root: str | Path) -> dict[str, Path]:
    docs_root = Path(root)
    return {
        normalize_doc_key(path, docs_root): path
        for path in iter_supported_files(docs_root)
    }


def diff_markdown_documents(
    previous_state: dict[str, Any],
    current_hashes: dict[str, str],
) -> MarkdownDocumentDiff:
    previous_docs = {
        str(key): dict(value)
        for key, value in previous_state.get("docs", {}).items()
        if isinstance(value, dict)
    }
    previous_keys = set(previous_docs)
    current_keys = set(current_hashes)
    added = current_keys - previous_keys
    deleted = previous_keys - current_keys
    changed = {
        key
        for key in current_keys & previous_keys
        if str(previous_docs[key].get("content_hash") or "") != current_hashes[key]
    }
    unchanged = (current_keys & previous_keys) - changed
    return MarkdownDocumentDiff(
        added=added,
        changed=changed,
        deleted=deleted,
        unchanged=unchanged,
    )


def load_source_manifest(input_dir: str | Path) -> dict[str, dict[str, str]]:
    """读取清洗阶段生成的路径映射清单。"""
    manifest_path = Path(input_dir) / SOURCE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("documents", []) if isinstance(payload, dict) else []
    manifest: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        relative_path = str(record.get("markdown_relative_path") or "")
        if relative_path:
            manifest[_normalize_manifest_key(relative_path)] = {
                str(key): str(value)
                for key, value in record.items()
                if value is not None
            }
    return manifest


def iter_supported_files(input_dir: str | Path, recursive: bool = True) -> Iterable[Path]:
    """列出当前加载器会处理的文件，便于调试输入范围。"""
    root = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _normalize_document_metadata(
    document: Document,
    root: Path,
    source_manifest: dict[str, dict[str, str]],
) -> Document:
    metadata = dict(document.metadata or {})
    markdown_path = _pick_source_path(document, metadata)
    suffix = Path(markdown_path).suffix.lower().lstrip(".")

    metadata.setdefault("doc_id", markdown_path)
    metadata.setdefault("file_name", Path(markdown_path).name)
    metadata.setdefault("file_type", "html" if suffix == "htm" else suffix)
    metadata.setdefault("cleaned_markdown_path", markdown_path)

    source_record = _pick_source_record(markdown_path, root, source_manifest)
    if source_record:
        metadata.update(
            {
                "source_doc_id": source_record.get("source_path", ""),
                "source_path": source_record.get("source_path", ""),
                "source_absolute_path": source_record.get("source_absolute_path", ""),
                "source_relative_path": source_record.get("source_relative_path", ""),
                "source_file_name": source_record.get("source_file_name", ""),
                "source_file_type": source_record.get("source_file_type", ""),
                "converted_html_path": source_record.get("converted_html_path", ""),
                "cleaned_markdown_path": source_record.get("markdown_path", markdown_path),
                "cleaned_markdown_relative_path": source_record.get("markdown_relative_path", ""),
            }
        )
    document.metadata = metadata
    return document


def _pick_source_path(document: Document, metadata: dict) -> str:
    for key in ("file_path", "filename", "doc_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(getattr(document, "doc_id", "") or getattr(document, "id_", ""))


def _pick_source_record(
    markdown_path: str,
    root: Path,
    source_manifest: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    for key in _manifest_lookup_keys(markdown_path, root):
        record = source_manifest.get(key)
        if record:
            return record
    return None


def _manifest_lookup_keys(markdown_path: str, root: Path) -> list[str]:
    path = Path(markdown_path)
    keys = [_normalize_manifest_key(markdown_path)]
    try:
        keys.append(_normalize_manifest_key(path.resolve().relative_to(root.resolve()).as_posix()))
    except ValueError:
        pass
    try:
        keys.append(_normalize_manifest_key(path.relative_to(root).as_posix()))
    except ValueError:
        pass
    return list(dict.fromkeys(keys))


def _normalize_manifest_key(path: str) -> str:
    return path.replace("\\", "/").strip().lower()
