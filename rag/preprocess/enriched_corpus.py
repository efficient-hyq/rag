"""预处理之后生成带图片说明的 Markdown，并按现有规则导出 chunk。"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.config.common import default_storage_dir, resolve_project_path
from rag.indexing.chunk_context import (
    DROPPED_METADATA_KEYS,
    archive_path_from_relative,
)
from rag.indexing.document_loader import load_documents
from rag.indexing.markdown_chunker import split_documents
from rag.preprocess.document_preprocessor import PreprocessConfig, preprocess_documents
from rag.preprocess.image_captions import (
    extract_chunk_images,
    inject_image_captions,
    iter_markdown_image_keys,
    load_image_captions,
    normalize_image_key,
)
from rag.shared.checkpoints import ProgressSnapshot, print_progress
from rag.shared.logging_utils import log_phase


logger = logging.getLogger("rag.preprocess.enriched")


@dataclass(frozen=True)
class PrepareChunksConfig:
    """带图片说明的切分导出配置。存储路径按项目根目录解析。"""

    input_dir: Path
    storage_dir: Path = field(default_factory=default_storage_dir)
    captions_path: Path | None = None
    skip_preprocess: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 100
    recursive: bool = True
    soffice_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_dir", resolve_project_path(self.storage_dir))
        if self.captions_path is not None:
            object.__setattr__(self, "captions_path", resolve_project_path(self.captions_path))

    @property
    def markdown_dir(self) -> Path:
        return self.storage_dir / "cleaned_markdown"

    @property
    def html_dir(self) -> Path:
        return self.storage_dir / "converted_html"

    @property
    def image_dir(self) -> Path:
        return self.storage_dir / "cleaned_assets"

    @property
    def enriched_dir(self) -> Path:
        return self.storage_dir / "enriched_markdown"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "prepared_chunks" / "chunks.jsonl"

    @property
    def report_path(self) -> Path:
        return self.storage_dir / "prepared_chunks" / "report.json"

    @property
    def resolved_captions_path(self) -> Path:
        if self.captions_path is not None:
            return self.captions_path
        return self.storage_dir / "image_captions.json"


@dataclass(frozen=True)
class PrepareChunksResult:
    source_count: int
    markdown_count: int
    chunk_count: int
    image_count: int
    captioned_image_count: int
    missing_caption_count: int
    enriched_dir: Path
    chunks_path: Path
    report_path: Path


def prepare_enriched_corpus(config: PrepareChunksConfig) -> PrepareChunksResult:
    """清洗原始文档，把图片说明写入正文，再按当前切分规则导出 chunk。

    标注、摘要、上下文嵌入和向量化不在这一步做，留给 build_index 读取 chunks.jsonl。
    """
    if not config.skip_preprocess:
        with log_phase(logger, "原始文档预处理", input_dir=str(config.input_dir)):
            preprocess_documents(
                PreprocessConfig(
                    input_dir=config.input_dir,
                    html_dir=config.html_dir,
                    markdown_dir=config.markdown_dir,
                    image_dir=config.image_dir,
                    recursive=config.recursive,
                    soffice_path=config.soffice_path,
                )
            )

    captions = load_image_captions(config.resolved_captions_path)
    with log_phase(logger, "写入图片说明", captions=len(captions)):
        export_stats = export_enriched_markdown(
            config.markdown_dir,
            config.enriched_dir,
            captions,
        )
    with log_phase(logger, "按现有规则切分", chunk_size=config.chunk_size):
        records = export_chunk_records(
            config.enriched_dir,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    write_chunk_records(config.chunks_path, records)
    report = _build_report(config, export_stats, records)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "切分导出完成 | markdown=%s | chunk=%s | 已配说明图片=%s | 缺说明图片=%s",
        export_stats["markdown_count"],
        len(records),
        export_stats["captioned_image_count"],
        export_stats["missing_caption_count"],
    )
    return PrepareChunksResult(
        source_count=int(export_stats["source_count"]),
        markdown_count=int(export_stats["markdown_count"]),
        chunk_count=len(records),
        image_count=int(export_stats["image_count"]),
        captioned_image_count=int(export_stats["captioned_image_count"]),
        missing_caption_count=int(export_stats["missing_caption_count"]),
        enriched_dir=config.enriched_dir,
        chunks_path=config.chunks_path,
        report_path=config.report_path,
    )


def export_enriched_markdown(
    markdown_dir: Path,
    enriched_dir: Path,
    captions: dict[str, str],
) -> dict[str, Any]:
    """按来源清单复制清洗结果，并在图片同一行末尾写入已有中文说明。"""
    markdown_root = Path(markdown_dir)
    manifest_path = markdown_root / "source_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少来源清单: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("documents", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise ValueError(f"来源清单格式不正确: {manifest_path}")

    output_root = Path(enriched_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rewritten: list[dict[str, Any]] = []
    missing: list[str] = []
    image_keys: list[str] = []
    total = len(records)
    for done, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        relative = str(record.get("markdown_relative_path") or "").replace("\\", "/")
        if not relative:
            continue
        source_md = markdown_root / relative
        if not source_md.is_file():
            raise FileNotFoundError(f"清单中的 Markdown 不存在: {source_md}")
        enriched_text, missing_keys = inject_image_captions(
            source_md.read_text(encoding="utf-8"),
            captions,
        )
        image_keys.extend(iter_markdown_image_keys(enriched_text))
        missing.extend(missing_keys)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(enriched_text, encoding="utf-8")
        rewritten.append(_rewrite_manifest_record(record, target, output_root))
        print_progress(ProgressSnapshot(stage="图片说明", total=total, done=done))

    manifest_out = output_root / "source_manifest.json"
    manifest_out.write_text(
        json.dumps({"documents": rewritten}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unique_images = list(dict.fromkeys(image_keys))
    unique_missing = list(dict.fromkeys(missing))
    return {
        "source_count": len(rewritten),
        "markdown_count": len(rewritten),
        "image_count": len(unique_images),
        "captioned_image_count": len(unique_images) - len(unique_missing),
        "missing_caption_count": len(unique_missing),
        "missing_captions": unique_missing,
        "unreferenced_assets": _unreferenced_assets(markdown_root.parent / "cleaned_assets", unique_images),
    }


def export_chunk_records(
    enriched_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """用当前 Markdown 切分器导出 chunk，并附上该片段里的图片说明。

    只保留预处理后的文档相对路径和归档目录，不写原始 doc/docx 路径。
    """
    root = Path(enriched_dir).resolve()
    documents = load_documents(root)
    _prepare_documents_for_chunking(documents, root)
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: list[dict[str, Any]] = []
    for node in nodes:
        metadata = dict(node.metadata or {})
        text = node.get_content()
        relative = str(metadata.get("markdown_relative_path") or "")
        records.append(
            {
                "node_id": str(getattr(node, "node_id", None) or getattr(node, "id_", "")),
                "doc_id": relative,
                "text": text,
                "heading_path": str(metadata.get("heading_path") or ""),
                "chunk_index": metadata.get("chunk_index"),
                "token_size": metadata.get("token_size"),
                "file_name": str(metadata.get("file_name") or Path(relative).name),
                "markdown_relative_path": relative,
                "archive_path": str(metadata.get("archive_path") or archive_path_from_relative(relative)),
                "images": extract_chunk_images(text),
            }
        )
    return records


def _prepare_documents_for_chunking(documents: list[Any], root: Path) -> None:
    """切分前把文档身份收成预处理相对路径，并丢掉原始文件路径。"""
    for document in documents:
        metadata = dict(document.metadata or {})
        relative = _relative_to_root(metadata.get("file_path") or metadata.get("doc_id"), root)
        metadata["doc_id"] = relative
        metadata["markdown_relative_path"] = relative
        metadata["archive_path"] = archive_path_from_relative(relative)
        metadata["file_name"] = str(metadata.get("file_name") or Path(relative).name)
        for key in DROPPED_METADATA_KEYS:
            metadata.pop(key, None)
        document.metadata = metadata


def write_chunk_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_manifest_record(record: dict[str, Any], markdown_path: Path, markdown_root: Path) -> dict[str, Any]:
    rewritten = dict(record)
    rewritten["markdown_path"] = str(markdown_path)
    rewritten["markdown_absolute_path"] = str(markdown_path.resolve())
    rewritten["markdown_relative_path"] = markdown_path.resolve().relative_to(markdown_root.resolve()).as_posix()
    return rewritten


def _relative_to_root(file_path: object, root: Path) -> str:
    if not file_path:
        return ""
    path = Path(str(file_path))
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _unreferenced_assets(image_dir: Path, referenced: list[str]) -> list[str]:
    if not image_dir.exists():
        return []
    referenced_keys = {normalize_image_key(item) for item in referenced}
    storage_root = image_dir.parent
    unused: list[str] = []
    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            continue
        relative = normalize_image_key(path.resolve().relative_to(storage_root.resolve()).as_posix())
        if relative not in referenced_keys:
            unused.append(relative)
    return sorted(unused)


def _build_report(
    config: PrepareChunksConfig,
    export_stats: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    image_chunks = sum(1 for record in records if record["images"])
    return {
        "input_dir": str(config.input_dir),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "enriched_dir": str(config.enriched_dir),
        "chunks_path": str(config.chunks_path),
        "captions_path": str(config.resolved_captions_path),
        "source_count": export_stats["source_count"],
        "markdown_count": export_stats["markdown_count"],
        "chunk_count": len(records),
        "chunks_with_images": image_chunks,
        "image_count": export_stats["image_count"],
        "captioned_image_count": export_stats["captioned_image_count"],
        "missing_caption_count": export_stats["missing_caption_count"],
        "missing_captions": export_stats["missing_captions"],
        "unreferenced_assets": export_stats["unreferenced_assets"],
        "note": "chunk 只保留预处理后的文档相对路径。标注、摘要、上下文嵌入和向量化由 build_index 读取 prepared_chunks/chunks.jsonl 完成。",
    }
