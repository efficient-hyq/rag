from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rag.config import (
    AnnotationConfig,
    BuildIndexConfig,
    ChunkingConfig,
    DependencyGraphConfig,
    EmbeddingConfig,
    LLMServiceConfig,
    PathConfig,
)
from rag.indexing.dependency_graph import DependencyAggregationResult
from rag.indexing.index_builder import build_offline_index
from rag.indexing.publication import INDEX_WORKSPACE_DIRNAME, resolve_final_index
from rag.indexing.semantic_annotator import AnnotationRunResult, SemanticAnnotator
from rag.retrieval.dependency_recall import DependencyRecallRouter
from rag.retrieval.retriever import HybridRetriever
from rag.shared.checkpoints import node_key


class FakeAnnotator:
    def annotate_nodes(self, nodes):
        for node in nodes:
            node.metadata.update(
                {
                    "summary": "支付接入说明",
                    "keywords": ["支付", "接入"],
                    "tags": ["支付"],
                    "type": "text",
                    "has_code": False,
                    "coherence": "high",
                    "dependency_topics": [],
                }
            )
        return AnnotationRunResult(nodes=nodes, failed_keys=[])


class FakeEmbedder:
    def get_text_embedding_batch(self, texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def get_text_embedding(self, text):
        return [float(len(text)), 1.0, 0.5]


class FakeDependencyAggregator:
    def aggregate_documents(self, nodes, doc_hashes, root_doc_dir):
        node_ids = [node_key(node) for node in nodes]
        doc_key = next(iter(doc_hashes))
        card = {
            "status": "success",
            "doc_content_hash": doc_hashes[doc_key],
            "expected_node_count": len(node_ids),
            "processed_node_count": len(node_ids),
            "model": "fake",
            "prompt_version": "fake-v1",
            "generated_at": "2026-08-04T00:00:00Z",
            "error": None,
            "discarded_edge_count": 0,
            "node_dependencies": {node_id: [] for node_id in node_ids},
        }
        return DependencyAggregationResult(documents={doc_key: card}, failed_docs=[])


class FailedDependencyAggregator(FakeDependencyAggregator):
    def aggregate_documents(self, nodes, doc_hashes, root_doc_dir):
        result = super().aggregate_documents(nodes, doc_hashes, root_doc_dir)
        doc_key = next(iter(result.documents))
        result.documents[doc_key]["status"] = "partial"
        result.documents[doc_key]["processed_node_count"] = 0
        return DependencyAggregationResult(documents=result.documents, failed_docs=[doc_key])


class CountingSemanticAnnotator(SemanticAnnotator):
    def __init__(self, prompt_version: str, summary: str) -> None:
        super().__init__(
            api_key="",
            client=object(),
            model="fake-annotator",
            max_workers=1,
            prompt="test",
            prompt_version=prompt_version,
        )
        self.summary = summary
        self.call_count = 0

    def annotate_text(self, text: str) -> dict:
        self.call_count += 1
        return {
            "summary": self.summary,
            "keywords": ["支付"],
            "tags": ["支付"],
            "type": "text",
            "has_code": False,
            "coherence": "high",
            "dependency_topics": [],
        }


class FlakyCheckpointEmbedder:
    cache_identity = "fake-checkpoint-embedding"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_once_text: str | None = None

    def get_text_embedding_batch(self, texts):
        self.calls.append(list(texts))
        if self.fail_once_text is not None and self.fail_once_text in texts:
            self.fail_once_text = None
            raise RuntimeError("模拟摘要向量失败")
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def get_text_embedding(self, text):
        return [float(len(text)), 1.0, 0.5]


class IndexPublicationBuilderTest(unittest.TestCase):
    def test_retry_reuses_new_annotation_and_completed_embeddings_after_prompt_rebuild_failure(self) -> None:
        root, docs_dir, storage_dir, config = prepare_build_case("payment.md")
        del root
        embedder = FlakyCheckpointEmbedder()
        old_config = replace(
            config,
            embedding=EmbeddingConfig(api_key="test-key"),
            annotation=AnnotationConfig(prompt_version="annotation_old"),
        )
        new_config = replace(
            old_config,
            annotation=AnnotationConfig(prompt_version="annotation_new"),
        )
        old_annotator = CountingSemanticAnnotator("annotation_old", "旧摘要")
        new_annotator = CountingSemanticAnnotator("annotation_new", "新摘要")

        with (
            mock.patch("rag.indexing.index_builder.SemanticAnnotator", return_value=old_annotator),
            mock.patch(
                "rag.indexing.index_builder.build_openai_compatible_embedder",
                return_value=embedder,
            ),
        ):
            build_offline_index(
                docs_dir=docs_dir,
                storage_dir=storage_dir,
                dependency_aggregator=FakeDependencyAggregator(),
                config=old_config,
            )
        published_manifest = (storage_dir / "manifest.json").read_bytes()

        embedder.fail_once_text = "新摘要"
        with (
            mock.patch("rag.indexing.index_builder.SemanticAnnotator", return_value=new_annotator),
            mock.patch(
                "rag.indexing.index_builder.build_openai_compatible_embedder",
                return_value=embedder,
            ),
        ):
            with self.assertRaises(RuntimeError):
                build_offline_index(
                    docs_dir=docs_dir,
                    storage_dir=storage_dir,
                    dependency_aggregator=FakeDependencyAggregator(),
                    config=new_config,
                )
            self.assertEqual((storage_dir / "manifest.json").read_bytes(), published_manifest)

            result = build_offline_index(
                docs_dir=docs_dir,
                storage_dir=storage_dir,
                dependency_aggregator=FakeDependencyAggregator(),
                config=new_config,
            )

        flattened_calls = [text for batch in embedder.calls for text in batch]
        self.assertEqual(result.node_count, 1)
        self.assertEqual(old_annotator.call_count, 1)
        self.assertEqual(new_annotator.call_count, 1)
        self.assertEqual(sum("调用支付接口" in text for text in flattened_calls), 1)
        self.assertEqual(flattened_calls.count("旧摘要"), 1)
        self.assertEqual(flattened_calls.count("新摘要"), 2)

    def test_legacy_topic_registry_version_does_not_rebuild_unchanged_documents(self) -> None:
        root, docs_dir, storage_dir, config = prepare_build_case("service.md")
        del root

        build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        workspace = storage_dir / INDEX_WORKSPACE_DIRNAME
        state_path = workspace / "checkpoints" / "document_index_state.json"
        legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
        legacy_state["topic_registry_version"] = "任意旧版本"
        state_path.write_text(json.dumps(legacy_state, ensure_ascii=False), encoding="utf-8")

        rebuilt = build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(rebuilt.node_count, 0)
        self.assertEqual(state["annotation_prompt_version"], "annotation_v4")
        self.assertEqual(state["dependency_prompt_version"], "dependency_graph_v3")
        self.assertNotIn("topic_registry_version", state)

    def test_build_failure_keeps_final_index_and_retry_publishes_workspace(self) -> None:
        root, docs_dir, storage_dir, config = prepare_build_case("payment.md")
        del root

        first = build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        final_dir = resolve_final_index(storage_dir)
        first_metadata = (final_dir / "metadata.json").read_bytes()
        first_manifest = (storage_dir / "manifest.json").read_bytes()

        second = build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        self.assertEqual(first.node_count, 1)
        self.assertEqual(second.node_count, 0)
        second_final_dir = resolve_final_index(storage_dir)
        self.assertNotEqual(second_final_dir, final_dir)
        final_dir = second_final_dir
        first_metadata = (final_dir / "metadata.json").read_bytes()
        first_manifest = (storage_dir / "manifest.json").read_bytes()

        retrieval = HybridRetriever(
            embedder=FakeEmbedder(),
            storage_dir=storage_dir,
            neighbor_enabled=False,
            dependency_router=DependencyRecallRouter(llm_enabled=False),
        ).retrieve("如何接入支付")
        self.assertEqual(len(retrieval.top_candidates), 1)

        (docs_dir / "payment.md").write_text("# 支付接入\n\n未完成内容。", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            build_offline_index(
                docs_dir=docs_dir,
                storage_dir=storage_dir,
                annotator=FakeAnnotator(),
                embedder=FakeEmbedder(),
                dependency_aggregator=FailedDependencyAggregator(),
                config=config,
            )
        self.assertEqual((final_dir / "metadata.json").read_bytes(), first_metadata)
        self.assertEqual((storage_dir / "manifest.json").read_bytes(), first_manifest)

        retried = build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        final_dir = resolve_final_index(storage_dir)
        published_metadata = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(retried.node_count, 1)
        self.assertTrue(any("未完成内容" in item["text"] for item in published_metadata.values()))

        (docs_dir / "payment.md").unlink()
        deleted = build_offline_index(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            annotator=FakeAnnotator(),
            embedder=FakeEmbedder(),
            dependency_aggregator=FakeDependencyAggregator(),
            config=config,
        )
        final_dir = resolve_final_index(storage_dir)
        cards = json.loads((final_dir / "dependency_cards.json").read_text(encoding="utf-8"))
        metadata = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(deleted.node_count, 0)
        self.assertEqual(cards["documents"], {})
        self.assertEqual(metadata, {})


def prepare_build_case(file_name: str):
    root = Path(tempfile.mkdtemp())
    docs_dir = root / "docs"
    storage_dir = root / "storage"
    docs_dir.mkdir()
    (docs_dir / file_name).write_text("# 支付接入\n\n调用支付接口。", encoding="utf-8")
    config = BuildIndexConfig(
        paths=PathConfig(docs_dir=docs_dir, storage_dir=storage_dir),
        llm=LLMServiceConfig(api_key=None),
        embedding=EmbeddingConfig(),
        chunking=ChunkingConfig(chunk_size=64, chunk_overlap=0),
        annotation=AnnotationConfig(),
        dependency=DependencyGraphConfig(vector_enabled=False),
    )
    return root, docs_dir, storage_dir, config


if __name__ == "__main__":
    unittest.main()
