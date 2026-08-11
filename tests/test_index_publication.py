from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from rag.indexing.publication import (
    INDEX_WORKSPACE_DIRNAME,
    PUBLISHED_INDEX_DIR_PREFIX,
    prepare_index_workspace,
    publish_index,
    resolve_final_index,
    validate_query_embedding_model,
)
from rag.indexing.storage_indexer import SimpleBM25Okapi, StoredBM25Index


class IndexPublicationTest(unittest.TestCase):
    def test_successful_workspace_is_published_to_manifest_final_directory(self) -> None:
        storage = Path(tempfile.mkdtemp())
        workspace = prepare_index_workspace(storage)
        write_valid_workspace(workspace)

        final_dir = publish_index(storage, workspace)

        self.assertEqual(workspace, storage / INDEX_WORKSPACE_DIRNAME)
        self.assertEqual(final_dir.parent, storage)
        self.assertTrue(final_dir.name.startswith(PUBLISHED_INDEX_DIR_PREFIX))
        self.assertEqual(resolve_final_index(storage), final_dir)
        manifest = json.loads((storage / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["final_index_dir"], final_dir.name)
        self.assertEqual(manifest["embedding_model"], "text-embedding-v4")
        self.assertTrue((workspace / "checkpoints" / "annotations.jsonl").exists())
        self.assertFalse((final_dir / "checkpoints").exists())

    def test_failed_workspace_does_not_replace_published_index(self) -> None:
        storage = Path(tempfile.mkdtemp())
        workspace = prepare_index_workspace(storage)
        write_valid_workspace(workspace, text="已发布内容")
        final_dir = publish_index(storage, workspace)
        published_metadata = (final_dir / "metadata.json").read_bytes()
        published_manifest = (storage / "manifest.json").read_bytes()

        (workspace / "metadata.json").write_text(
            json.dumps({"unexpected-node": {"text": "未完成内容"}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            publish_index(storage, workspace)

        self.assertEqual((final_dir / "metadata.json").read_bytes(), published_metadata)
        self.assertEqual((storage / "manifest.json").read_bytes(), published_manifest)

    def test_prepare_reuses_existing_workspace_checkpoints(self) -> None:
        storage = Path(tempfile.mkdtemp())
        first = prepare_index_workspace(storage)
        checkpoint = first / "checkpoints" / "annotations.jsonl"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('{"key":"node-1","status":"success"}\n', encoding="utf-8")

        second = prepare_index_workspace(storage)

        self.assertEqual(first, second)
        self.assertEqual(second, storage / INDEX_WORKSPACE_DIRNAME)
        self.assertTrue(checkpoint.exists())

    def test_query_embedding_model_must_match_published_index(self) -> None:
        storage = Path(tempfile.mkdtemp())
        workspace = prepare_index_workspace(storage)
        write_valid_workspace(workspace)
        publish_index(storage, workspace)

        self.assertEqual(
            validate_query_embedding_model(storage, "text-embedding-v4"),
            "text-embedding-v4",
        )
        with self.assertRaisesRegex(ValueError, "查询 Embedding 模型与已发布索引不一致"):
            validate_query_embedding_model(storage, "text-embedding-v3")

    def test_query_rejects_legacy_manifest_without_embedding_model(self) -> None:
        storage = Path(tempfile.mkdtemp())
        (storage / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "index_manifest_v1",
                    "final_index_dir": "published_index-old",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "缺少 embedding_model"):
            validate_query_embedding_model(storage, "text-embedding-v4")


def write_valid_workspace(root: Path, text: str = "流程") -> None:
    (root / "chroma").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    metadata = {"node-1": {"text": text, "doc_id": "payment.md"}}
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    state = {
        "docs": {"payment.md": {"content_hash": "hash", "node_ids": ["node-1"]}},
    }
    (root / "checkpoints" / "document_index_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (root / "checkpoints" / "annotations.jsonl").write_text(
        '{"key":"node-1","status":"success"}\n', encoding="utf-8"
    )
    (root / "checkpoints" / "manifest.json").write_text(
        json.dumps({"embedding_model": "text-embedding-v4"}),
        encoding="utf-8",
    )
    cards = {
        "schema_version": "dependency_graph_v4",
        "documents": {
            "payment.md": {
                "status": "success",
                "doc_content_hash": "hash",
                "expected_node_count": 1,
                "processed_node_count": 1,
                "node_dependencies": {"node-1": []},
            }
        },
    }
    (root / "dependency_cards.json").write_text(json.dumps(cards), encoding="utf-8")
    bm25 = StoredBM25Index(
        node_ids=["node-1"],
        tokenized_corpus=[[text]],
        bm25=SimpleBM25Okapi([[text]]),
    )
    with (root / "bm25.pkl").open("wb") as file:
        pickle.dump(bm25, file)


if __name__ == "__main__":
    unittest.main()
