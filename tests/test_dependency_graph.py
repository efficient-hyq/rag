from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag.indexing.dependency_graph import (
    build_aggregation_input,
    load_dependency_prompt,
    validate_dependency_document,
)


class DependencyGraphTest(unittest.TestCase):
    def test_dependency_prompt_does_not_require_topic(self) -> None:
        prompt = load_dependency_prompt()

        self.assertIn("topic 是可选", prompt)
        self.assertNotIn("受控目录", prompt)

    def test_aggregation_input_contains_complete_text(self) -> None:
        node = SimpleNamespace(
            id_="node-1",
            text="# 标题\n\n必须保留完整限定条件。",
            metadata={"chunk_index": 0, "heading_path": "标题", "summary": "摘要"},
        )

        payload = build_aggregation_input([node])

        self.assertEqual(payload[0]["text"], node.text)
        self.assertEqual(payload[0]["heading_path"], "标题")

    def test_validation_fills_empty_nodes_and_discards_invalid_edges(self) -> None:
        raw = {
            "node_dependencies": {
                "node-1": [
                    {
                        "node_id": "node-2",
                        "topic": "请求验签",
                        "relation_type": "prerequisite",
                        "required": True,
                        "confidence": 0.9,
                        "reason": "调用前必须签名",
                    },
                    {
                        "node_id": "node-2",
                        "relation_type": "constraint",
                        "required": True,
                        "confidence": 0.8,
                        "reason": "重复目标边",
                    },
                    {
                        "node_id": "node-1",
                        "relation_type": "constraint",
                        "required": True,
                        "confidence": 0.8,
                        "reason": "自身依赖",
                    },
                ],
                "node-2": [],
            }
        }

        card = validate_dependency_document(raw, ["node-1", "node-2"], "hash", "model", "v1")

        self.assertEqual(card["status"], "success")
        self.assertEqual(set(card["node_dependencies"]), {"node-1", "node-2"})
        self.assertEqual(len(card["node_dependencies"]["node-1"]), 1)
        self.assertEqual(
            card["node_dependencies"]["node-1"][0]["topic"],
            "请求验签",
        )
        self.assertEqual(card["node_dependencies"]["node-2"], [])
        self.assertEqual(card["discarded_edge_count"], 2)

    def test_edge_without_topic_remains_valid(self) -> None:
        raw = {
            "node_dependencies": {
                "node-1": [
                    {
                        "node_id": "node-2",
                        "relation_type": "prerequisite",
                        "required": True,
                        "confidence": 0.9,
                        "reason": "接入前必须完成配置",
                    }
                ],
                "node-2": [],
            }
        }

        card = validate_dependency_document(raw, ["node-1", "node-2"], "hash", "model", "v1")

        self.assertEqual(card["status"], "success")
        self.assertEqual(len(card["node_dependencies"]["node-1"]), 1)
        self.assertNotIn("topic", card["node_dependencies"]["node-1"][0])

    def test_missing_source_node_marks_document_partial(self) -> None:
        card = validate_dependency_document(
            {"node_dependencies": {"node-1": []}},
            ["node-1", "node-2"],
            "hash",
            "model",
            "v1",
        )

        self.assertEqual(card["status"], "partial")
        self.assertEqual(card["processed_node_count"], 1)
        self.assertEqual(card["node_dependencies"]["node-2"], [])


if __name__ == "__main__":
    unittest.main()
