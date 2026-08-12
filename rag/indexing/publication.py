"""离线索引工作区校验与最终结果发布。"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.indexing.dependency_graph import DEPENDENCY_SCHEMA_VERSION


INDEX_WORKSPACE_DIRNAME = "index_workspace"
PUBLISHED_INDEX_DIR_PREFIX = "published_index-"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "index_manifest_v1"

WORKSPACE_ARTIFACTS = (
    "chroma",
    "bm25.pkl",
    "metadata.json",
    "checkpoints",
    "chunk_previews",
    "dependency_cards.json",
)

PUBLISHED_ARTIFACTS = (
    "chroma",
    "bm25.pkl",
    "metadata.json",
    "dependency_cards.json",
)


def prepare_index_workspace(storage_dir: str | Path) -> Path:
    """返回固定构建工作区；首次使用时迁移根目录中的旧式索引产物。"""
    storage_root = Path(storage_dir)
    storage_root.mkdir(parents=True, exist_ok=True)
    workspace = storage_root / INDEX_WORKSPACE_DIRNAME
    if workspace.exists():
        if not workspace.is_dir():
            raise NotADirectoryError(f"索引工作区不是目录: {workspace}")
        return workspace

    workspace.mkdir()
    if not (storage_root / MANIFEST_FILENAME).exists():
        _copy_artifacts(storage_root, workspace, WORKSPACE_ARTIFACTS)
    return workspace


def resolve_final_index(storage_dir: str | Path) -> Path:
    """严格按根 manifest.json 解析查询使用的最终索引目录。"""
    storage_root = Path(storage_dir)
    manifest_path = storage_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"索引发布清单不存在: {manifest_path}")
    payload = _load_json_object(manifest_path)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest.json schema_version 不匹配")

    final_dir_value = payload.get("final_index_dir")
    if not isinstance(final_dir_value, str) or not final_dir_value.strip():
        raise ValueError("manifest.json 缺少 final_index_dir")
    relative_path = Path(final_dir_value.strip())
    if (
        relative_path.is_absolute()
        or len(relative_path.parts) != 1
        or relative_path.name in {".", ".."}
    ):
        raise ValueError("manifest.json 的 final_index_dir 必须是存储根目录下的单层相对目录")

    final_dir = storage_root / relative_path
    if not final_dir.is_dir():
        raise FileNotFoundError(f"最终索引目录不存在: {final_dir}")
    return final_dir


def validate_query_embedding_model(storage_dir: str | Path, query_model: str) -> str:
    """校验查询编码模型与已发布索引的 Embedding 模型严格一致。"""
    manifest_path = Path(storage_dir) / MANIFEST_FILENAME
    payload = _load_json_object(manifest_path)
    indexed_model = str(payload.get("embedding_model") or "").strip()
    current_model = str(query_model or "").strip()
    if not indexed_model:
        raise ValueError(
            "索引发布清单缺少 embedding_model，无法确认向量空间；请重新构建并发布索引"
        )
    if not current_model:
        raise ValueError("查询 Embedding 模型不能为空")
    if current_model != indexed_model:
        raise ValueError(
            "查询 Embedding 模型与已发布索引不一致: "
            f"query={current_model}, index={indexed_model}；请使用建库模型或全量重建索引"
        )
    return indexed_model


def publish_index(storage_dir: str | Path, workspace_dir: str | Path) -> Path:
    """校验固定工作区，通过后创建最终目录并切换唯一有效清单。"""
    storage_root = Path(storage_dir)
    workspace = Path(workspace_dir)
    expected_workspace = storage_root / INDEX_WORKSPACE_DIRNAME
    if workspace.resolve() != expected_workspace.resolve():
        raise ValueError(f"只能发布固定索引工作区: {expected_workspace}")

    validate_index_workspace(workspace)
    build_manifest = _load_json_object(workspace / "checkpoints" / "manifest.json")
    embedding_model = str(build_manifest.get("embedding_model") or "").strip()
    if not embedding_model:
        raise ValueError("工作区构建清单缺少 embedding_model")

    previous_final = (
        resolve_final_index(storage_root)
        if (storage_root / MANIFEST_FILENAME).exists()
        else None
    )
    token = uuid.uuid4().hex
    candidate = storage_root / f".{PUBLISHED_INDEX_DIR_PREFIX}publishing-{token}"
    manifest_temporary = storage_root / f".{MANIFEST_FILENAME}.{token}.tmp"
    final_dir = storage_root / f"{PUBLISHED_INDEX_DIR_PREFIX}{token}"

    candidate.mkdir(parents=False, exist_ok=False)
    try:
        _copy_artifacts(workspace, candidate, PUBLISHED_ARTIFACTS)
        _validate_published_copy(candidate, workspace)
        manifest_temporary.write_text(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "final_index_dir": final_dir.name,
                    "embedding_model": embedding_model,
                    "published_at": _utc_now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            candidate.rename(final_dir)
            manifest_temporary.replace(storage_root / MANIFEST_FILENAME)
        except Exception:
            if final_dir.exists():
                _remove_path(final_dir)
            raise
    finally:
        if candidate.exists():
            _remove_path(candidate)
        if manifest_temporary.exists():
            manifest_temporary.unlink()

    _cleanup_obsolete_publications(storage_root, final_dir, previous_final)
    return final_dir


def validate_index_workspace(workspace_dir: str | Path) -> None:
    """校验 metadata、BM25、文档状态和依赖图的 node 引用一致性。"""
    root = Path(workspace_dir)
    metadata = _load_json_object(root / "metadata.json")
    cards = _load_json_object(root / "dependency_cards.json")
    state = _load_json_object(root / "checkpoints" / "document_index_state.json")
    build_manifest = _load_json_object(root / "checkpoints" / "manifest.json")

    if not str(build_manifest.get("embedding_model") or "").strip():
        raise ValueError("工作区构建清单缺少 embedding_model")

    if cards.get("schema_version") != DEPENDENCY_SCHEMA_VERSION:
        raise ValueError("依赖图 schema_version 不匹配")

    documents = cards.get("documents")
    state_docs = state.get("docs")
    if not isinstance(documents, dict) or not isinstance(state_docs, dict):
        raise ValueError("依赖图或文档状态格式非法")
    if set(documents) != set(state_docs):
        raise ValueError("依赖图文档集合与文档状态不一致")

    metadata_ids = set(metadata)
    state_node_ids: set[str] = set()
    for doc_key, record in state_docs.items():
        if not isinstance(record, dict):
            raise ValueError(f"文档状态非法: {doc_key}")
        node_ids = {str(item) for item in record.get("node_ids", []) if item}
        state_node_ids.update(node_ids)
        card = documents.get(doc_key)
        if not isinstance(card, dict) or card.get("status") != "success":
            raise ValueError(f"文档依赖图未成功: {doc_key}")
        if card.get("doc_content_hash") != record.get("content_hash"):
            raise ValueError(f"文档依赖图内容哈希不一致: {doc_key}")
        dependencies = card.get("node_dependencies")
        if not isinstance(dependencies, dict) or set(dependencies) != node_ids:
            raise ValueError(f"文档依赖图 node 覆盖不完整: {doc_key}")
        if (
            card.get("expected_node_count") != len(node_ids)
            or card.get("processed_node_count") != len(node_ids)
        ):
            raise ValueError(f"文档依赖图 node 数量不一致: {doc_key}")
        if not node_ids.issubset(metadata_ids):
            raise ValueError(f"文档状态存在 metadata 缺失 node: {doc_key}")
        for source_id, edges in dependencies.items():
            if source_id not in node_ids or not isinstance(edges, list):
                raise ValueError(f"文档依赖图 source 非法: {doc_key}")
            for edge in edges:
                if not isinstance(edge, dict) or edge.get("node_id") not in node_ids:
                    raise ValueError(f"文档依赖图 target 非法: {doc_key}")

    if state_node_ids != metadata_ids:
        raise ValueError("文档状态 node 集合与 metadata 不一致")

    bm25_path = root / "bm25.pkl"
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 索引不存在: {bm25_path}")
    with bm25_path.open("rb") as file:
        bm25_payload: Any = pickle.load(file)
    if set(str(item) for item in getattr(bm25_payload, "node_ids", [])) != metadata_ids:
        raise ValueError("BM25 node 集合与 metadata 不一致")
    if metadata_ids and not (root / "chroma").is_dir():
        raise FileNotFoundError("Chroma 索引目录不存在")


def _validate_published_copy(candidate: Path, workspace: Path) -> None:
    for name in ("bm25.pkl", "metadata.json", "dependency_cards.json"):
        if not (candidate / name).is_file():
            raise FileNotFoundError(f"最终索引产物缺失: {candidate / name}")
    metadata = _load_json_object(candidate / "metadata.json")
    workspace_metadata = _load_json_object(workspace / "metadata.json")
    if metadata != workspace_metadata:
        raise ValueError("最终目录 metadata 与工作区不一致")
    if metadata and not (candidate / "chroma").is_dir():
        raise FileNotFoundError("最终目录 Chroma 索引缺失")


def _copy_artifacts(source_dir: Path, target_dir: Path, artifact_names: tuple[str, ...]) -> None:
    for name in artifact_names:
        source = source_dir / name
        target = target_dir / name
        if not source.exists() or source.resolve() == target.resolve():
            continue
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _cleanup_obsolete_publications(
    storage_root: Path,
    current_final: Path,
    previous_final: Path | None,
) -> None:
    logger = logging.getLogger("rag.pipeline")
    candidates = set(storage_root.glob(f"{PUBLISHED_INDEX_DIR_PREFIX}*"))
    if previous_final is not None:
        candidates.add(previous_final)
    for path in sorted(candidates):
        if path == current_final or not path.is_dir():
            continue
        try:
            _remove_path(path)
        except OSError:
            logger.warning("旧最终索引仍被查询进程占用，暂缓清理 | path=%s", path)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"索引产物不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"索引产物顶层必须是对象: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
