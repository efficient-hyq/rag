from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag.indexing.storage_indexer import MultiRouteIndexer, StoredBM25Index


class FakeCollection:
    def __init__(self) -> None:
        self.upserts = []
        self.deletes = []

    def upsert(self, **payload) -> None:
        self.upserts.append(payload)

    def delete(self, ids: list[str]) -> None:
        self.deletes.append(list(ids))


class FakeClient:
    def __init__(self) -> None:
        self.collections = {
            "content_vec": FakeCollection(),
            "summary_vec": FakeCollection(),
        }

    def get_or_create_collection(self, name: str):
        return self.collections[name]


class IncrementalStorageIndexerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.storage_dir = Path(tempfile.mkdtemp())
        self.client = FakeClient()
        self.indexer = MultiRouteIndexer(
            self.storage_dir,
            chroma_client=self.client,
            tokenizer=lambda text: text.split(),
        )

    def test_delete_nodes_removes_from_both_collections(self) -> None:
        self.indexer.delete_nodes({"n1", "n2"})
        self.assertEqual(len(self.client.collections["content_vec"].deletes), 1)
        self.assertEqual(sorted(self.client.collections["content_vec"].deletes[0]), ["n1", "n2"])
        self.assertEqual(len(self.client.collections["summary_vec"].deletes), 1)
        self.assertEqual(sorted(self.client.collections["summary_vec"].deletes[0]), ["n1", "n2"])

    def test_write_metadata_shards_and_compat_snapshot(self) -> None:
        node = SimpleNamespace(
            id_="node-a",
            text="chunk text",
            metadata={
                "doc_id": "docs/a.md",
                "file_name": "a.md",
                "chunk_index": 0,
                "summary": "sum",
            },
        )
        self.indexer.write_metadata_shards([node], root_doc_dir=Path("docs"))
        self.indexer.rebuild_metadata_snapshot()

        shard_files = list((self.storage_dir / "metadata_docs").glob("*.json"))
        self.assertEqual(len(shard_files), 1)
        metadata = json.loads((self.storage_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("node-a", metadata)
        self.assertEqual(metadata["node-a"]["text"], "chunk text")

    def test_rebuild_bm25_from_metadata_snapshot_uses_all_current_nodes(self) -> None:
        (self.storage_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "node-a": {"text": "alpha beta", "summary": "sum a", "keywords": ["ka"]},
                    "node-b": {"text": "beta gamma", "summary": "sum b", "tags": ["tb"]},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.indexer.rebuild_bm25_from_metadata_snapshot()

        with self.indexer.bm25_path.open("rb") as file:
            payload = pickle.load(file)
        self.assertIsInstance(payload, StoredBM25Index)
        self.assertEqual(payload.node_ids, ["node-a", "node-b"])
