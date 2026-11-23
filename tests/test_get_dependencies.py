from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.agent import tools


class GetDependenciesTests(unittest.TestCase):
    def test_get_dependencies_returns_topic_and_reason_without_text(self) -> None:
        long_text = "依赖正文" * 100
        retriever = SimpleNamespace(
            metadata_index=SimpleNamespace(
                metadata_by_id={
                    "node-1": {"text": "片段正文"},
                    "dep-1": {
                        "text": long_text,
                        "summary": "验签摘要",
                        "file_name": "回调.md",
                        "heading_path": "回调 > 验签",
                    },
                }
            ),
            dependency_graph=SimpleNamespace(
                get_required_edges=lambda node_id: [
                    {
                        "node_id": "dep-1",
                        "topic": "验签",
                        "reason": "回调必须验签",
                        "confidence": 0.9,
                        "text": "依赖全文不应该返回",
                    }
                ]
            ),
        )
        with patch.object(tools, "_get_retriever", return_value=retriever):
            payload = json.loads(tools.get_dependencies(" node-1 "))

        self.assertTrue(payload["found"])
        dependency = payload["dependencies"][0]
        self.assertEqual(dependency["node_id"], "dep-1")
        self.assertEqual(dependency["topic"], "验签")
        self.assertEqual(dependency["reason"], "回调必须验签")
        self.assertEqual(dependency["summary"], "验签摘要")
        self.assertEqual(len(dependency["snippet"]), 200)
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("片段正文", rendered)
        self.assertNotIn("依赖全文不应该返回", rendered)
        self.assertNotIn(long_text, rendered)

    def test_get_dependencies_returns_empty_list_when_node_has_no_edges(self) -> None:
        retriever = SimpleNamespace(
            metadata_index=SimpleNamespace(metadata_by_id={"node-1": {}}),
            dependency_graph=SimpleNamespace(get_required_edges=lambda node_id: []),
        )
        with patch.object(tools, "_get_retriever", return_value=retriever):
            payload = json.loads(tools.get_dependencies("node-1"))

        self.assertTrue(payload["found"])
        self.assertEqual(payload["dependencies"], [])

    def test_get_dependencies_reports_missing_node(self) -> None:
        retriever = SimpleNamespace(
            metadata_index=SimpleNamespace(metadata_by_id={}),
            dependency_graph=SimpleNamespace(get_required_edges=lambda node_id: [{"node_id": "dep-1"}]),
        )
        with patch.object(tools, "_get_retriever", return_value=retriever):
            payload = json.loads(tools.get_dependencies("missing"))

        self.assertFalse(payload["found"])
        self.assertEqual(payload["node_id"], "missing")


if __name__ == "__main__":
    unittest.main()
