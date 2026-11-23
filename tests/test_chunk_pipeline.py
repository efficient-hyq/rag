from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rag.config import BuildIndexConfig
from rag.cli.prepare_chunks import parse_args
from rag.config.common import default_storage_dir, project_root, resolve_project_path
from rag.indexing.chunk_context import (
    CONTEXT_VERSION,
    build_context_text,
    normalize_chunk_record,
)
from rag.indexing.dependency_graph import DependencyAggregationResult, group_nodes_by_doc
from rag.indexing.index_builder import build_offline_index
from rag.indexing.rebuild_decision import CHUNK_STATE_SCHEMA, compute_chunk_rebuild_decision
from rag.indexing.storage_indexer import IndexResult
from rag.retrieval.ranking import apply_archive_affinity
from rag.shared.checkpoints import node_key


GOOGLE_PATH = "支付订阅服务/Google订阅/说明.md"
GATEWAY_PATH = "支付订阅服务/支付网关/说明.md"


class ChunkContextTests(unittest.TestCase):
    def test_relative_storage_stays_under_project_root_when_cwd_changes(self) -> None:
        with TemporaryDirectory() as temp:
            previous = Path.cwd()
            os.chdir(temp)
            try:
                resolved = resolve_project_path("storage")
            finally:
                os.chdir(previous)

        self.assertEqual(resolved, (project_root() / "storage").resolve())
        self.assertFalse(str(resolved).startswith(str(Path(temp).resolve())))

    def test_prepare_chunks_defaults_stay_under_project_storage(self) -> None:
        with TemporaryDirectory() as temp:
            previous = Path.cwd()
            os.chdir(temp)
            try:
                args = parse_args([])
            finally:
                os.chdir(previous)

        storage = default_storage_dir()
        self.assertEqual(Path(args.input_dir), storage / "wiki-export")
        self.assertEqual(Path(args.storage_dir), storage)
        self.assertTrue(Path(args.input_dir).is_absolute())
        self.assertTrue(Path(args.storage_dir).is_absolute())
        self.assertFalse(str(Path(args.storage_dir)).startswith(str(Path(temp).resolve())))

    def test_context_text_keeps_archive_and_google_alias(self) -> None:
        text = build_context_text(
            "开通后可取消订阅。",
            {
                "archive_path": "支付订阅服务/Google订阅",
                "file_name": "说明.md",
                "heading_path": "取消",
            },
        )
        self.assertIn("归档：支付订阅服务/Google订阅", text)
        self.assertIn("别名：谷歌", text)
        self.assertIn("开通后可取消订阅。", text)
        self.assertNotIn("source_path", text)

    def test_normalize_rejects_original_absolute_path(self) -> None:
        with self.assertRaises(ValueError):
            normalize_chunk_record({"markdown_relative_path": r"D:\raw\说明.md", "text": "正文"})

    def test_google_query_demotes_payment_gateway(self) -> None:
        gateway = SimpleNamespace(
            node_id="gateway",
            metadata={"archive_path": "支付订阅服务/支付网关"},
            final_score=0.2,
        )
        google = SimpleNamespace(
            node_id="google",
            metadata={"archive_path": "支付订阅服务/Google订阅"},
            final_score=0.05,
        )
        ordered = apply_archive_affinity("谷歌订阅怎么取消", [gateway, google])
        self.assertEqual(ordered[0].node_id, "google")
        self.assertGreater(google.final_score, gateway.final_score)

    def test_generic_query_does_not_change_scores(self) -> None:
        gateway = SimpleNamespace(
            node_id="gateway",
            metadata={"archive_path": "支付订阅服务/支付网关"},
            final_score=0.2,
        )
        google = SimpleNamespace(
            node_id="google",
            metadata={"archive_path": "支付订阅服务/Google订阅"},
            final_score=0.05,
        )
        ordered = apply_archive_affinity("怎么退款", [gateway, google])
        self.assertEqual([item.node_id for item in ordered], ["gateway", "google"])
        self.assertEqual(gateway.final_score, 0.2)
        self.assertEqual(google.final_score, 0.05)


class ChunkRebuildDecisionTests(unittest.TestCase):
    def test_added_replaced_and_deleted_are_chunk_scoped(self) -> None:
        with TemporaryDirectory() as temp:
            decision = compute_chunk_rebuild_decision(
                {"a": "h1", "b": "h2-new"},
                {"doc": "doc-hash"},
                {"a": "doc", "b": "doc"},
                {
                    "schema_version": CHUNK_STATE_SCHEMA,
                    "context_version": CONTEXT_VERSION,
                    "chunks": {
                        "a": {"content_hash": "h1", "doc_key": "doc"},
                        "b": {"content_hash": "h2-old", "doc_key": "doc"},
                        "c": {"content_hash": "h3", "doc_key": "doc"},
                    },
                    "docs": {"doc": {"content_hash": "old"}},
                    "annotation_prompt_version": "annotation_v4",
                    "annotation_prompt_hash": "same",
                    "dependency_prompt_version": "dependency_graph_v4",
                    "dependency_prompt_hash": "same",
                    "embedding_model": "text-embedding-v4",
                },
                annotation_prompt_version="annotation_v4",
                annotation_prompt_hash="same",
                dependency_prompt_version="dependency_graph_v4",
                dependency_prompt_hash="same",
                embedding_model="text-embedding-v4",
                storage_root=Path(temp),
            )

        self.assertEqual(decision.unchanged, {"a"})
        self.assertEqual(decision.replaced, {"b"})
        self.assertEqual(decision.deleted, {"c"})
        self.assertIn("b", decision.need_reannotate)
        self.assertNotIn("a", decision.need_reannotate)
        self.assertIn("doc", decision.need_redependency_docs)

    def test_prompt_change_reannotates_unchanged_chunks(self) -> None:
        with TemporaryDirectory() as temp:
            decision = compute_chunk_rebuild_decision(
                {"a": "h1"},
                {"doc": "doc-hash"},
                {"a": "doc"},
                {
                    "schema_version": CHUNK_STATE_SCHEMA,
                    "context_version": CONTEXT_VERSION,
                    "chunks": {"a": {"content_hash": "h1", "doc_key": "doc"}},
                    "docs": {"doc": {"content_hash": "doc-hash"}},
                    "annotation_prompt_version": "annotation_v4",
                    "annotation_prompt_hash": "old",
                    "dependency_prompt_version": "dependency_graph_v4",
                    "dependency_prompt_hash": "same",
                    "embedding_model": "text-embedding-v4",
                },
                annotation_prompt_version="annotation_v4",
                annotation_prompt_hash="new",
                dependency_prompt_version="dependency_graph_v4",
                dependency_prompt_hash="same",
                embedding_model="text-embedding-v4",
                storage_root=Path(temp),
            )

        self.assertEqual(decision.need_reannotate, {"a"})
        self.assertEqual(decision.need_reembed, {"a"})
        self.assertEqual(decision.replaced, set())

    def test_situating_context_reannotates_and_reembeds(self) -> None:
        previous = {
            "schema_version": CHUNK_STATE_SCHEMA,
            "context_version": CONTEXT_VERSION,
            "chunks": {"a": {"content_hash": "h1", "doc_key": "doc"}},
            "docs": {"doc": {"content_hash": "doc-hash"}},
            "annotation_prompt_version": "annotation_v4",
            "annotation_prompt_hash": "same",
            "dependency_prompt_version": "dependency_graph_v4",
            "dependency_prompt_hash": "same",
            "embedding_model": "text-embedding-v4",
        }
        with TemporaryDirectory() as temp:
            introduced = compute_chunk_rebuild_decision(
                {"a": "h1"},
                {"doc": "doc-hash"},
                {"a": "doc"},
                previous,
                annotation_prompt_version="annotation_v4",
                annotation_prompt_hash="same",
                dependency_prompt_version="dependency_graph_v4",
                dependency_prompt_hash="same",
                embedding_model="text-embedding-v4",
                storage_root=Path(temp),
                situating_prompt_version="chunk_context_v1",
                situating_prompt_hash="situating-hash",
                situating_doc_hashes={"doc": "whole-new"},
            )
            changed_document = compute_chunk_rebuild_decision(
                {"a": "h1"},
                {"doc": "doc-hash"},
                {"a": "doc"},
                {
                    **previous,
                    "situating_prompt_version": "chunk_context_v1",
                    "situating_prompt_hash": "situating-hash",
                    "situating_doc_hashes": {"doc": "whole-old"},
                },
                annotation_prompt_version="annotation_v4",
                annotation_prompt_hash="same",
                dependency_prompt_version="dependency_graph_v4",
                dependency_prompt_hash="same",
                embedding_model="text-embedding-v4",
                storage_root=Path(temp),
                situating_prompt_version="chunk_context_v1",
                situating_prompt_hash="situating-hash",
                situating_doc_hashes={"doc": "whole-new"},
            )

        self.assertEqual(introduced.need_reannotate, {"a"})
        self.assertEqual(introduced.need_reembed, {"a"})
        self.assertEqual(changed_document.need_reannotate, {"a"})
        self.assertEqual(changed_document.need_reembed, {"a"})


class ChunkIndexTests(unittest.TestCase):
    def test_build_index_uses_prepared_chunks_and_skips_unchanged(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            chunks_file = root / "prepared_chunks" / "chunks.jsonl"
            self._write_chunks(chunks_file, [_google_chunk(), _gateway_chunk()])
            annotator = _FakeAnnotator()
            embedder = _FakeEmbedder()
            dependency = _FakeDependency()
            indexer = _FakeIndexer(root)
            config = replace(
                BuildIndexConfig.from_env(),
                dependency=replace(BuildIndexConfig.from_env().dependency, vector_enabled=False),
            )
            situating = _FakeSituating()

            first = build_offline_index(
                chunks_file,
                root,
                annotator=annotator,
                embedder=embedder,
                indexer=indexer,
                dependency_aggregator=dependency,
                situating_generator=situating,
                config=config,
            )
            second = build_offline_index(
                chunks_file,
                root,
                annotator=annotator,
                embedder=embedder,
                indexer=indexer,
                dependency_aggregator=dependency,
                situating_generator=situating,
                config=config,
            )

            self.assertEqual(first.node_count, 2)
            self.assertEqual(second.node_count, 0)
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(dependency.calls, 1)
            self.assertEqual(len(indexer.indexed), 1)
            content_batch = embedder.batches[0]
            google_context = next(text for text in content_batch if "开通订阅" in text)
            gateway_context = next(text for text in content_batch if "验签" in text)
            self.assertIn("归档：支付订阅服务/Google订阅", google_context)
            self.assertIn("这一片段位于对应业务文档中。", google_context)
            self.assertIn("谷歌", google_context)
            self.assertTrue(annotator.saw_context_before_annotation)
            self.assertEqual(situating.calls, 1)
            self.assertIn("归档：支付订阅服务/支付网关", gateway_context)
            self.assertNotIn("谷歌", gateway_context)
            indexed_text = {node.node_id: node.text for node in indexer.indexed[0]}
            self.assertNotIn("归档：", indexed_text["google-node"])
            self.assertNotIn("这一片段位于对应业务文档中。", indexed_text["google-node"])

            self._write_chunks(chunks_file, [_google_chunk()])
            third = build_offline_index(
                chunks_file,
                root,
                annotator=annotator,
                embedder=embedder,
                indexer=indexer,
                dependency_aggregator=dependency,
                situating_generator=situating,
                config=config,
            )

        self.assertEqual(third.node_count, 1)
        self.assertIn("gateway-node", set().union(*indexer.deleted))
        self.assertEqual([node.node_id for node in indexer.indexed[-1]], ["google-node"])
        self.assertEqual(annotator.calls, 1)

    def _write_chunks(self, path: Path, records: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


class _FakeSituating:
    def __init__(self) -> None:
        self.calls = 0

    def apply(
        self,
        nodes: list[object],
        documents: dict[str, str],
        document_hashes: dict[str, str],
        checkpoint: object | None = None,
    ) -> None:
        self.calls += 1
        for node in nodes:
            metadata = dict(getattr(node, "metadata", {}) or {})
            metadata["situating_context"] = "这一片段位于对应业务文档中。"
            node.metadata = metadata


class _FakeAnnotator:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_context_before_annotation = False

    def annotate_nodes(self, nodes: list[object], checkpoint: object | None = None) -> SimpleNamespace:
        self.calls += 1
        self.saw_context_before_annotation = all(
            str((getattr(node, "metadata", {}) or {}).get("situating_context") or "")
            for node in nodes
        )
        for node in nodes:
            metadata = dict(getattr(node, "metadata", {}) or {})
            metadata["summary"] = "摘要"
            metadata["keywords"] = ["关键词"]
            node.metadata = metadata
        return SimpleNamespace(nodes=nodes, failed_count=0)


class _FakeEmbedder:
    cache_identity = "fake"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0, 0.2] for _ in texts]

    def get_text_embedding(self, text: str) -> list[float]:
        return [1.0, 0.2]


class _FakeDependency:
    def __init__(self) -> None:
        self.calls = 0

    def aggregate_documents(
        self,
        nodes: list[object],
        doc_hashes: dict[str, str],
        root: Path,
        checkpoint: object | None = None,
    ) -> DependencyAggregationResult:
        self.calls += 1
        documents: dict[str, dict[str, object]] = {}
        for doc_key, doc_nodes in group_nodes_by_doc(nodes, Path(root)).items():
            documents[doc_key] = {
                "status": "success",
                "doc_content_hash": doc_hashes.get(doc_key, ""),
                "expected_node_count": len(doc_nodes),
                "processed_node_count": len(doc_nodes),
                "node_dependencies": {node_key(node): [] for node in doc_nodes},
            }
        return DependencyAggregationResult(documents=documents, failed_docs=[])


class _FakeIndexer:
    def __init__(self, root: Path) -> None:
        self.deleted: list[set[str]] = []
        self.indexed: list[list[object]] = []
        self.bm25_path = root / "bm25.pkl"
        self.metadata_path = root / "metadata.json"

    def delete_nodes(self, node_ids: set[str]) -> None:
        self.deleted.append(set(node_ids))

    def delete_dependency_nodes(self, node_ids: set[str]) -> None:
        return None

    def index(
        self,
        nodes: list[object],
        content_embeddings: list[list[float]],
        summary_embeddings: list[list[float]],
        root_doc_dir: Path | None = None,
    ) -> IndexResult:
        self.indexed.append(list(nodes))
        return IndexResult(
            node_count=len(nodes),
            content_collection="content_vec",
            summary_collection="summary_vec",
            bm25_path=self.bm25_path,
            metadata_path=self.metadata_path,
        )

    def index_dependency_vectors(self, records: list[dict[str, object]], embeddings: list[list[float]]) -> None:
        return None


def _google_chunk() -> dict[str, object]:
    return {
        "node_id": "google-node",
        "doc_id": GOOGLE_PATH,
        "text": "开通订阅后在后台取消。",
        "heading_path": "取消",
        "chunk_index": 0,
        "token_size": 8,
        "file_name": "说明.md",
        "markdown_relative_path": GOOGLE_PATH,
        "archive_path": "支付订阅服务/Google订阅",
        "images": [],
    }


def _gateway_chunk() -> dict[str, object]:
    return {
        "node_id": "gateway-node",
        "doc_id": GATEWAY_PATH,
        "text": "网关回调需要验签。",
        "heading_path": "验签",
        "chunk_index": 0,
        "token_size": 8,
        "file_name": "说明.md",
        "markdown_relative_path": GATEWAY_PATH,
        "archive_path": "支付订阅服务/支付网关",
        "images": [],
    }


if __name__ == "__main__":
    unittest.main()
