from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag.indexing.document_loader import compute_document_content_hash
from rag.indexing.index_builder import build_offline_index
from rag.indexing.storage_indexer import IndexResult
from rag.shared.checkpoints import CheckpointStore


class IncrementalIndexBuilderTest(unittest.TestCase):
    def test_only_changed_documents_are_reloaded_and_reindexed(self) -> None:
        docs_dir = Path(tempfile.mkdtemp())
        storage_dir = Path(tempfile.mkdtemp())
        (docs_dir / "a.md").write_text("# A\nchanged\n", encoding="utf-8")
        (docs_dir / "b.md").write_text("# B\nkeep\n", encoding="utf-8")
        checkpoint = CheckpointStore(storage_dir)
        checkpoint.save_document_index_state(
            {
                "docs": {
                    "a.md": {
                        "content_hash": "old",
                        "node_ids": ["old-a"],
                        "updated_at": "2026-06-05T10:00:00+08:00",
                    },
                    "b.md": {
                        "content_hash": compute_document_content_hash(docs_dir / "b.md"),
                        "node_ids": ["keep-b"],
                        "updated_at": "2026-06-05T10:00:00+08:00",
                    },
                    "c.md": {
                        "content_hash": "gone",
                        "node_ids": ["old-c"],
                        "updated_at": "2026-06-05T10:00:00+08:00",
                    },
                }
            }
        )
        fake_indexer = mock.Mock()

        with mock.patch("rag.indexing.index_builder.load_documents_from_files") as load_docs:
            load_docs.return_value = []
            with self.assertRaises(RuntimeError):
                build_offline_index(
                    docs_dir=docs_dir,
                    storage_dir=storage_dir,
                    annotator=object(),
                    embedder=object(),
                    indexer=fake_indexer,
                )

        loaded_files = [Path(path) for path in load_docs.call_args.args[0]]
        self.assertEqual(sorted(path.name for path in loaded_files), ["a.md"])
        fake_indexer.delete_nodes.assert_called_once_with({"old-a", "old-c"})
        fake_indexer.remove_metadata_shard.assert_any_call("a.md")
        fake_indexer.remove_metadata_shard.assert_any_call("c.md")

    def test_unchanged_documents_skip_rebuild_and_refresh_snapshots(self) -> None:
        docs_dir = Path(tempfile.mkdtemp())
        storage_dir = Path(tempfile.mkdtemp())
        (docs_dir / "a.md").write_text("# A\nkeep\n", encoding="utf-8")
        checkpoint = CheckpointStore(storage_dir)
        checkpoint.save_document_index_state(
            {
                "docs": {
                    "a.md": {
                        "content_hash": compute_document_content_hash(docs_dir / "a.md"),
                        "node_ids": ["keep-a"],
                        "updated_at": "2026-06-05T10:00:00+08:00",
                    }
                }
            }
        )
        fake_indexer = mock.Mock()
        fake_indexer.bm25_path = storage_dir / "bm25.pkl"
        fake_indexer.metadata_path = storage_dir / "metadata.json"

        with mock.patch("rag.indexing.index_builder.load_documents_from_files") as load_docs:
            result = build_offline_index(
                docs_dir=docs_dir,
                storage_dir=storage_dir,
                annotator=object(),
                embedder=object(),
                indexer=fake_indexer,
            )

        load_docs.assert_not_called()
        fake_indexer.delete_nodes.assert_not_called()
        fake_indexer.rebuild_metadata_snapshot.assert_called_once_with()
        fake_indexer.rebuild_bm25_from_metadata_snapshot.assert_called_once_with()
        self.assertEqual(
            result,
            IndexResult(
                node_count=0,
                content_collection="content_vec",
                summary_collection="summary_vec",
                bm25_path=fake_indexer.bm25_path,
                metadata_path=fake_indexer.metadata_path,
            ),
        )

    def test_rebuild_uses_scanned_file_path_instead_of_normalized_key(self) -> None:
        docs_dir = Path(tempfile.mkdtemp())
        storage_dir = Path(tempfile.mkdtemp())
        (docs_dir / "A.md").write_text("# A\nchanged\n", encoding="utf-8")

        with mock.patch("rag.indexing.index_builder.load_documents_from_files") as load_docs:
            load_docs.return_value = []
            with self.assertRaises(RuntimeError):
                build_offline_index(
                    docs_dir=docs_dir,
                    storage_dir=storage_dir,
                    annotator=object(),
                    embedder=object(),
                    indexer=mock.Mock(),
                )

        loaded_files = [Path(path) for path in load_docs.call_args.args[0]]
        self.assertEqual([path.name for path in loaded_files], ["A.md"])
