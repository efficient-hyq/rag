from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.indexing.dependency_graph import DEPENDENCY_SCHEMA_VERSION
from rag.indexing.publication import MANIFEST_FILENAME, publish_index, resolve_final_index
from rag.indexing.storage_indexer import SimpleBM25Okapi, StoredBM25Index


class PublicationTests(unittest.TestCase):
    def test_publish_replaces_manifest_when_recorded_directory_is_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            (root / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": "index_manifest_v1",
                        "final_index_dir": "published_index-missing",
                        "embedding_model": "old-model",
                        "published_at": "2026-11-23T00:00:00Z",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            final_dir = publish_index(root, workspace)

            self.assertTrue(final_dir.is_dir())
            self.assertEqual(resolve_final_index(root), final_dir)
            manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_index_dir"], final_dir.name)
            self.assertEqual(manifest["embedding_model"], "test-model")
            self.assertFalse((root / "published_index-missing").exists())

    def test_publish_keeps_new_directory_and_removes_previous_one(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            previous = root / "published_index-previous"
            previous.mkdir()
            (previous / "metadata.json").write_text("{}", encoding="utf-8")
            (root / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": "index_manifest_v1",
                        "final_index_dir": previous.name,
                        "embedding_model": "old-model",
                        "published_at": "2026-11-23T00:00:00Z",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            final_dir = publish_index(root, workspace)

            self.assertTrue(final_dir.is_dir())
            self.assertFalse(previous.exists())
            self.assertEqual(resolve_final_index(root), final_dir)


def _workspace(root: Path) -> Path:
    workspace = root / "index_workspace"
    checkpoints = workspace / "checkpoints"
    checkpoints.mkdir(parents=True)
    (workspace / "metadata.json").write_text("{}", encoding="utf-8")
    (workspace / "dependency_cards.json").write_text(
        json.dumps({"schema_version": DEPENDENCY_SCHEMA_VERSION, "documents": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (checkpoints / "document_index_state.json").write_text(
        json.dumps({"docs": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (checkpoints / "manifest.json").write_text(
        json.dumps({"embedding_model": "test-model"}, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = StoredBM25Index(node_ids=[], tokenized_corpus=[], bm25=SimpleBM25Okapi([]))
    with (workspace / "bm25.pkl").open("wb") as file:
        pickle.dump(payload, file)
    return workspace


if __name__ == "__main__":
    unittest.main()
