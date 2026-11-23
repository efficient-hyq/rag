from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rag.agent import tools
from rag.retrieval.retriever import MetadataIndex


def _grep_chunk(relative_path: str, chunk_index: int, text: str) -> dict[str, object]:
    return {
        "text": text,
        "summary": text,
        "file_name": Path(relative_path).name,
        "heading_path": str(chunk_index),
        "cleaned_markdown_relative_path": relative_path,
        "doc_id": relative_path,
        "chunk_index": chunk_index,
    }


class ReadNeighborsTests(unittest.TestCase):
    def test_read_neighbors_returns_summaries_without_full_text(self) -> None:
        index = MetadataIndex(
            {
                "a": {"doc_id": "doc", "chunk_index": 0, "text": "甲" * 300, "summary": "摘要甲", "file_name": "指引.md", "heading_path": "一"},
                "b": {"doc_id": "doc", "chunk_index": 1, "text": "乙" * 300, "summary": "摘要乙", "file_name": "指引.md", "heading_path": "二"},
                "c": {"doc_id": "doc", "chunk_index": 2, "text": "丙" * 300, "summary": "摘要丙", "file_name": "指引.md", "heading_path": "三"},
            }
        )
        with patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)):
            payload = json.loads(tools.read_neighbors("b", radius=1))

        self.assertTrue(payload["found"])
        self.assertEqual([item["node_id"] for item in payload["neighbors"]], ["a", "c"])
        self.assertEqual([item["relation"] for item in payload["neighbors"]], ["previous", "next"])
        self.assertEqual(payload["neighbors"][0]["summary"], "摘要甲")
        self.assertEqual(len(payload["neighbors"][0]["snippet"]), 200)
        self.assertNotIn("甲" * 300, json.dumps(payload, ensure_ascii=False))

    def test_read_neighbors_reports_missing_node(self) -> None:
        index = MetadataIndex({})
        with patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)):
            payload = json.loads(tools.read_neighbors("missing"))
        self.assertFalse(payload["found"])


class SystemPromptTests(unittest.TestCase):
    def test_system_prompt_keeps_role_goal_and_retrieval_boundary(self) -> None:
        prompt = tools.SYSTEM_PROMPT
        self.assertIn("你是企业知识库问答助手", prompt)
        for heading in ("目标：", "可用手段：", "回应顺序：", "硬边界："):
            self.assertIn(heading, prompt)
        self.assertIn("检索量和是否继续由你判断", prompt)
        self.assertIn("当前结果已经够回答就停止", prompt)
        for detail in ("next_offset", "omitted_count", "max_results", "chunk_index", "top_k"):
            self.assertNotIn(detail, prompt)


class GrepDocsTests(unittest.TestCase):
    def test_grep_docs_returns_match_snippet_and_node_id(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp)
            markdown = docs_dir / "支付订阅服务" / "接口.md"
            markdown.parent.mkdir()
            markdown.write_text("前文" * 100 + "\nlinkedPurchaseToken 需要单独处理\n" + "后文" * 100, encoding="utf-8")
            index = MetadataIndex(
                {
                    "node-1": {
                        "text": "linkedPurchaseToken 需要单独处理",
                        "summary": "令牌处理",
                        "file_name": "接口.md",
                        "heading_path": "订阅 > 令牌",
                        "cleaned_markdown_relative_path": "支付订阅服务/接口.md",
                        "doc_id": "doc",
                        "chunk_index": 0,
                    }
                }
            )
            with (
                patch.object(tools, "_get_docs_dir", return_value=docs_dir),
                patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)),
            ):
                payload = json.loads(tools.grep_docs("linkedPurchaseToken", max_results=20))

        self.assertEqual(payload["omitted_count"], 0)
        self.assertIsNone(payload["next_offset"])
        self.assertEqual(len(payload["matches"]), 1)
        match = payload["matches"][0]
        self.assertEqual(match["node_id"], "node-1")
        self.assertEqual(match["summary"], "令牌处理")
        self.assertIn("linkedPurchaseToken", match["snippet"])
        self.assertEqual(match["file_path"], str(markdown.resolve()))
        self.assertTrue(Path(match["file_path"]).is_absolute())
        self.assertLessEqual(len(match["snippet"]), 200)
        self.assertNotIn("前文" * 100, match["snippet"])

    def test_grep_docs_returns_empty_list_when_pattern_is_absent(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp)
            (docs_dir / "接口.md").write_text("没有目标字段", encoding="utf-8")
            index = MetadataIndex({})
            with (
                patch.object(tools, "_get_docs_dir", return_value=docs_dir),
                patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)),
            ):
                payload = json.loads(tools.grep_docs("linkedPurchaseToken"))
        self.assertEqual(payload, {"matches": [], "omitted_count": 0, "next_offset": None})

    def test_grep_docs_returns_every_matching_chunk_in_chunk_index_order(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp)
            markdown = docs_dir / "接口.md"
            markdown.write_text("文件正文", encoding="utf-8")
            index = MetadataIndex(
                {
                    "later": {
                        "text": "后段 linkedPurchaseToken 说明",
                        "summary": "后段",
                        "file_name": "接口.md",
                        "heading_path": "后",
                        "cleaned_markdown_relative_path": "接口.md",
                        "doc_id": "doc",
                        "chunk_index": 2,
                    },
                    "earlier": {
                        "text": "前段 linkedPurchaseToken 说明",
                        "summary": "前段",
                        "file_name": "接口.md",
                        "heading_path": "前",
                        "cleaned_markdown_relative_path": "接口.md",
                        "doc_id": "doc",
                        "chunk_index": 0,
                    },
                    "middle": {
                        "text": "中段没有目标词",
                        "summary": "中段",
                        "file_name": "接口.md",
                        "heading_path": "中",
                        "cleaned_markdown_relative_path": "接口.md",
                        "doc_id": "doc",
                        "chunk_index": 1,
                    },
                }
            )
            with (
                patch.object(tools, "_get_docs_dir", return_value=docs_dir),
                patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)),
            ):
                payload = json.loads(tools.grep_docs("linkedPurchaseToken", max_results=1))

        self.assertEqual([item["node_id"] for item in payload["matches"]], ["earlier"])
        self.assertEqual(payload["matches"][0]["chunk_index"], 0)
        self.assertEqual(payload["matches"][0]["snippet"], "前段 linkedPurchaseToken 说明")
        self.assertEqual(payload["omitted_count"], 1)
        self.assertEqual(payload["next_offset"], 1)

    def test_grep_docs_pages_across_documents(self) -> None:
        with TemporaryDirectory() as temp:
            docs_dir = Path(temp)
            for name in ("a.md", "b.md"):
                (docs_dir / name).write_text(name, encoding="utf-8")
            index = MetadataIndex(
                {
                    "a-later": _grep_chunk("a.md", 1, "后段 token"),
                    "a-earlier": _grep_chunk("a.md", 0, "前段 token"),
                    "b-only": _grep_chunk("b.md", 0, "另一篇 token"),
                }
            )
            with (
                patch.object(tools, "_get_docs_dir", return_value=docs_dir),
                patch.object(tools, "_get_retriever", return_value=SimpleNamespace(metadata_index=index)),
            ):
                first = json.loads(tools.grep_docs("token", max_results=2))
                second = json.loads(tools.grep_docs("token", max_results=2, offset=first["next_offset"]))

        self.assertEqual([item["node_id"] for item in first["matches"]], ["a-earlier", "b-only"])
        self.assertEqual(first["omitted_count"], 1)
        self.assertEqual(first["next_offset"], 2)
        self.assertEqual([item["node_id"] for item in second["matches"]], ["a-later"])
        self.assertEqual(second["omitted_count"], 0)
        self.assertIsNone(second["next_offset"])

    def test_grep_docs_rejects_empty_pattern(self) -> None:
        payload = json.loads(tools.grep_docs("   "))
        self.assertEqual(payload["error"], "pattern 不能为空")


if __name__ == "__main__":
    unittest.main()
