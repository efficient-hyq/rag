from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.indexing.document_loader import (
    compute_document_content_hash,
    diff_markdown_documents,
    load_documents_from_files,
)


class IncrementalDocumentLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        (self.root / "a.md").write_text("# A\nalpha\n", encoding="utf-8")
        (self.root / "b.md").write_text("# B\nbeta\n", encoding="utf-8")

    def test_compute_document_content_hash_changes_with_file_content(self) -> None:
        original = compute_document_content_hash(self.root / "a.md")
        (self.root / "a.md").write_text("# A\nchanged\n", encoding="utf-8")
        changed = compute_document_content_hash(self.root / "a.md")
        self.assertNotEqual(original, changed)

    def test_diff_markdown_documents_returns_added_changed_deleted_and_unchanged(self) -> None:
        previous = {
            "docs": {
                "a.md": {
                    "content_hash": "old-hash",
                    "node_ids": ["n1"],
                    "updated_at": "2026-06-05T10:00:00+08:00",
                },
                "c.md": {
                    "content_hash": "hash-c",
                    "node_ids": ["n3"],
                    "updated_at": "2026-06-05T10:00:00+08:00",
                },
            }
        }
        current = {
            "a.md": "new-hash",
            "b.md": compute_document_content_hash(self.root / "b.md"),
        }
        diff = diff_markdown_documents(previous, current)
        self.assertEqual(diff.added, {"b.md"})
        self.assertEqual(diff.changed, {"a.md"})
        self.assertEqual(diff.deleted, {"c.md"})
        self.assertEqual(diff.unchanged, set())

    def test_load_documents_from_files_only_reads_selected_markdown(self) -> None:
        documents = load_documents_from_files([self.root / "b.md"], self.root)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["file_name"], "b.md")
