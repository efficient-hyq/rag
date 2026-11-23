from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.agent import tools
from rag.retrieval.retriever import HybridRetriever, RetrievalCandidate, RetrievedRouteHit


class SearchSnippetsTests(unittest.TestCase):
    def test_search_snippets_only_uses_original_query_and_rule_rerank(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.embedder = SimpleNamespace(get_text_embedding=lambda text: [0.1, 0.2])
        retriever.metadata_index = SimpleNamespace(
            metadata_by_id={
                "node-a": {"text": "甲" * 50, "file_name": "a.md", "summary": "摘要甲", "heading_path": "甲"},
                "node-b": {"text": "乙" * 50, "file_name": "b.md", "summary": "摘要乙", "heading_path": "乙"},
            }
        )
        retriever.rule_reranker = SimpleNamespace(rerank=lambda candidates: list(reversed(candidates)))
        seen: dict[str, object] = {}

        def retrieve_for_query(query: str, query_embedding: list[float], is_rewritten: bool) -> list[RetrievedRouteHit]:
            seen["query"] = query
            seen["embedding"] = query_embedding
            seen["is_rewritten"] = is_rewritten
            return [
                RetrievedRouteHit("node-a", "content_vec", 1, 0.2, query, False),
                RetrievedRouteHit("node-b", "bm25", 1, 1.0, query, False),
            ]

        retriever._retrieve_for_query = retrieve_for_query  # type: ignore[method-assign]

        selected = retriever.search_snippets("谷歌订阅回调", top_k=1)

        self.assertEqual(seen["query"], "谷歌订阅回调")
        self.assertEqual(seen["is_rewritten"], False)
        self.assertEqual([item.node_id for item in selected], ["node-b"])

    def test_search_knowledge_base_returns_short_snippet_json(self) -> None:
        full_text = "全文" * 300
        candidate = RetrievalCandidate(
            node_id="node-1",
            text=full_text,
            metadata={
                "file_name": "支付流程.md",
                "heading_path": "支付 > 回调",
                "summary": "回调验签说明",
            },
        )

        class FakeRetriever:
            def search_snippets(self, query: str, top_k: int) -> list[RetrievalCandidate]:
                self.query = query
                self.top_k = top_k
                return [candidate]

        fake = FakeRetriever()
        with patch.object(tools, "_get_retriever", return_value=fake):
            payload = json.loads(tools.search_knowledge_base("  谷歌订阅  ", top_k=20))

        self.assertEqual(fake.query, "谷歌订阅")
        self.assertEqual(fake.top_k, 8)
        self.assertEqual(payload[0]["node_id"], "node-1")
        self.assertEqual(payload[0]["file_name"], "支付流程.md")
        self.assertEqual(len(payload[0]["snippet"]), 200)
        self.assertNotIn("全文" * 101, payload[0]["snippet"])

    def test_search_knowledge_base_rejects_empty_query(self) -> None:
        payload = json.loads(tools.search_knowledge_base("   "))
        self.assertEqual(payload["error"], "query 不能为空")


if __name__ == "__main__":
    unittest.main()
