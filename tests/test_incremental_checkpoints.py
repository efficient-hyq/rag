from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.shared.checkpoints import CheckpointStore


class IncrementalCheckpointStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.storage_dir = Path(tempfile.mkdtemp())
        self.store = CheckpointStore(self.storage_dir)

    def test_save_and_load_document_index_state(self) -> None:
        payload = {
            "docs": {
                "a/b.md": {
                    "content_hash": "hash-1",
                    "node_ids": ["n1", "n2"],
                    "updated_at": "2026-06-05T10:00:00+08:00",
                }
            }
        }

        self.store.save_document_index_state(payload)

        self.assertEqual(self.store.load_document_index_state(), payload)

    def test_remove_node_records_cleans_all_checkpoint_files(self) -> None:
        self.store.upsert_chunk_records(
            [
                {
                    "node_id": "old-node",
                    "text_hash": "hash-old",
                    "metadata": {"doc_id": "doc-a.md"},
                    "text": "old text",
                },
                {
                    "node_id": "keep-node",
                    "text_hash": "hash-keep",
                    "metadata": {"doc_id": "doc-b.md"},
                    "text": "keep text",
                },
            ]
        )
        self.store.append_annotation(
            "old-node|annotation|qwen3.6-plus|annotation_v2",
            {"summary": "old"},
        )
        self.store.append_annotation(
            "keep-node|annotation|qwen3.6-plus|annotation_v2",
            {"summary": "keep"},
        )
        self.store.append_embedding(
            "content",
            "old-node|embedding|content|model",
            [0.1, 0.2],
        )
        self.store.append_embedding(
            "content",
            "keep-node|embedding|content|model",
            [0.3, 0.4],
        )
        self.store.append_embedding(
            "summary",
            "old-node|embedding|summary|model|summary:abc",
            [0.5, 0.6],
        )
        self.store.append_embedding(
            "summary",
            "keep-node|embedding|summary|model|summary:def",
            [0.7, 0.8],
        )

        self.store.remove_node_records({"old-node"})

        chunk_records = self.store.load_chunk_records()
        self.assertNotIn("old-node", chunk_records)
        self.assertIn("keep-node", chunk_records)
        self.assertNotIn(
            "old-node|annotation|qwen3.6-plus|annotation_v2",
            self.store.load_raw_records(self.store.annotations_path),
        )
        self.assertNotIn(
            "old-node|embedding|content|model",
            self.store.load_raw_records(self.store.content_embeddings_path),
        )
        self.assertNotIn(
            "old-node|embedding|summary|model|summary:abc",
            self.store.load_raw_records(self.store.summary_embeddings_path),
        )

    def test_load_llm_checkpoints_ignores_model_in_legacy_keys(self) -> None:
        self.store.append_annotation(
            "node-1|annotation|deepseek-v4-flash|annotation_v4",
            {"summary": "已完成标注"},
        )
        self.store.append_dependency_card(
            "doc.md|hash-1|deepseek-v4-flash|dependency_graph_v3",
            {
                "status": "success",
                "model": "deepseek-v4-flash",
                "node_dependencies": {"node-1": []},
            },
        )

        self.assertEqual(
            self.store.load_annotations()["node-1|annotation|annotation_v4"]["summary"],
            "已完成标注",
        )
        card = self.store.load_dependency_cards()["doc.md|hash-1|dependency_graph_v3"]
        self.assertEqual(card["model"], "deepseek-v4-flash")

    def test_migrate_llm_checkpoints_keeps_model_as_metadata(self) -> None:
        self.store.append_annotation(
            "node-1|annotation|deepseek-v4-flash|annotation_v4",
            {"summary": "已完成标注"},
        )
        self.store.append_dependency_card(
            "doc.md|hash-1|deepseek-v4-flash|dependency_graph_v3",
            {
                "status": "success",
                "model": "deepseek-v4-flash",
                "node_dependencies": {"node-1": []},
            },
        )

        counts = self.store.migrate_model_agnostic_records()

        annotations = self.store.load_raw_records(self.store.annotations_path)
        dependencies = self.store.load_raw_records(self.store.dependency_cards_path)
        self.assertEqual(counts, {"annotations": 1, "dependency_cards": 1})
        self.assertEqual(
            annotations["node-1|annotation|annotation_v4"]["model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(
            dependencies["doc.md|hash-1|dependency_graph_v3"]["model"],
            "deepseek-v4-flash",
        )

    def test_migrate_prefers_success_when_models_have_conflicting_records(self) -> None:
        self.store.append_annotation(
            "node-1|annotation|model-success|annotation_v4",
            {"summary": "可复用结果"},
            model="model-success",
        )
        self.store.append_annotation(
            "node-1|annotation|model-failed|annotation_v4",
            {"summary": ""},
            status="failed",
            model="model-failed",
        )

        self.store.migrate_model_agnostic_records()

        record = self.store.load_raw_records(self.store.annotations_path)[
            "node-1|annotation|annotation_v4"
        ]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["model"], "model-success")
