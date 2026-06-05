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
