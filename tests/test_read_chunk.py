from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rag.agent.tools import resolve_docs_dir, resolve_storage_dir
from rag.agent import tools


class ReadChunkTests(unittest.TestCase):
    def test_read_chunk_returns_indexed_text_and_safe_relative_path(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp) / "cleaned_markdown"
            markdown = docs_dir / "支付订阅服务" / "指引.md"
            markdown.parent.mkdir(parents=True)
            markdown.write_text("整篇文档不应该被当作片段返回", encoding="utf-8")
            metadata = {
                "text": "片段正文",
                "file_name": "指引.md",
                "heading_path": "支付 > 指引",
                "chunk_index": 2,
                "cleaned_markdown_relative_path": "支付订阅服务/指引.md",
                "file_path": r"storage\cleaned_markdown\支付订阅服务\指引.md",
                "doc_id": r"storage\cleaned_markdown\支付订阅服务\指引.md",
                "source_absolute_path": r"C:\Users\heyuqin\Desktop\RAG_DATA\指引.doc",
            }
            payload = self._read(docs_dir, metadata)

        self.assertTrue(payload["found"])
        self.assertEqual(payload["text"], "片段正文")
        self.assertEqual(payload["file_path"], str(markdown.resolve()))
        self.assertTrue(Path(payload["file_path"]).is_absolute())
        self.assertEqual(payload["file_name"], "指引.md")
        self.assertEqual(payload["chunk_index"], 2)
        self.assertNotIn("RAG_DATA", json.dumps(payload, ensure_ascii=False))

    def test_read_chunk_recovers_path_stored_relative_to_workspace(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp) / "cleaned_markdown"
            markdown = docs_dir / "支付订阅服务" / "指引.md"
            markdown.parent.mkdir(parents=True)
            markdown.write_text("文件正文", encoding="utf-8")
            metadata = {
                "text": "片段正文",
                "file_name": "指引.md",
                "file_path": r"storage\cleaned_markdown\支付订阅服务\指引.md",
                "source_absolute_path": r"C:\Users\heyuqin\Desktop\RAG_DATA\指引.doc",
            }
            payload = self._read(docs_dir, metadata)

        self.assertEqual(payload["text"], "片段正文")
        self.assertEqual(payload["file_path"], str(markdown.resolve()))
        self.assertTrue(Path(payload["file_path"]).is_absolute())

    def test_read_chunk_rejects_path_escape_and_original_doc(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp) / "docs"
            docs_dir.mkdir()
            outside = Path(temp) / "secret.md"
            outside.write_text("secret", encoding="utf-8")
            metadata = {
                "text": "片段正文",
                "cleaned_markdown_relative_path": "../secret.md",
                "file_path": str(outside),
                "source_absolute_path": str(outside),
            }
            payload = self._read(docs_dir, metadata)

        self.assertEqual(payload["text"], "片段正文")
        self.assertEqual(payload["file_path"], "")
        self.assertNotIn("secret.md", json.dumps(payload, ensure_ascii=False))

    def test_read_chunk_reports_missing_node(self) -> None:
        fake = SimpleNamespace(metadata_index=SimpleNamespace(metadata_by_id={}))
        with patch.object(tools, "_get_retriever", return_value=fake):
            payload = json.loads(tools.read_chunk("missing"))
        self.assertFalse(payload["found"])
        self.assertEqual(payload["node_id"], "missing")

    def test_storage_dir_stays_on_given_directory_without_manifest(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            default = root / "storage"
            default.mkdir()
            (default / "cleaned_markdown").mkdir()

            selected = resolve_storage_dir(None, default)
            docs = resolve_docs_dir(None, selected, root / "missing-docs")

        self.assertEqual(selected, default)
        self.assertEqual(docs, default / "cleaned_markdown")

    def test_docs_dir_prefers_enriched_markdown(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            storage = root / "storage"
            (storage / "enriched_markdown").mkdir(parents=True)
            (storage / "cleaned_markdown").mkdir()
            docs = resolve_docs_dir(None, storage, root / "missing-docs")
        self.assertEqual(docs, storage / "enriched_markdown")

    def test_explicit_storage_dir_is_kept(self) -> None:
        with TemporaryDirectory() as temp:
            configured = Path(temp) / "custom"
            selected = resolve_storage_dir(str(configured), Path(temp) / "default")
        self.assertEqual(selected, configured)

    def test_read_chunk_rejects_empty_node_id(self) -> None:
        payload = json.loads(tools.read_chunk("   "))
        self.assertFalse(payload["found"])
        self.assertEqual(payload["error"], "node_id 不能为空")

    def _read(self, docs_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
        fake = SimpleNamespace(metadata_index=SimpleNamespace(metadata_by_id={"node-1": metadata}))
        with (
            patch.object(tools, "_get_retriever", return_value=fake),
            patch.object(tools, "_get_docs_dir", return_value=docs_dir),
        ):
            return json.loads(tools.read_chunk(" node-1 "))


if __name__ == "__main__":
    unittest.main()
