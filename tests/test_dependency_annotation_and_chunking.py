from __future__ import annotations

import unittest

from llama_index.core.schema import Document

from rag.indexing.markdown_chunker import split_documents
from rag.indexing.semantic_annotator import load_annotation_prompt, normalize_annotation


class DependencyAnnotationAndChunkingTest(unittest.TestCase):
    def test_split_documents_generates_heading_path(self) -> None:
        document = Document(
            text="# 支付接入\n\n总体说明。\n\n## 回调处理\n\n回调处理说明。",
            metadata={"doc_id": "payment.md"},
        )

        nodes = split_documents([document], chunk_size=3, chunk_overlap=0)

        paths = [str(node.metadata.get("heading_path") or "") for node in nodes]
        self.assertIn("支付接入", paths)
        self.assertIn("支付接入 > 回调处理", paths)
        self.assertTrue(all("heading_path" in node.metadata for node in nodes))

    def test_annotation_keeps_free_form_unique_topics(self) -> None:
        annotation = normalize_annotation(
            {
                "dependency_topics": [
                    "回调幂等",
                    " 请求验签 ",
                    "回调幂等",
                    "支付渠道差错处理",
                ]
            }
        )

        self.assertEqual(
            annotation["dependency_topics"],
            ["回调幂等", "请求验签", "支付渠道差错处理"],
        )

    def test_prompt_allows_llm_to_label_topics_freely(self) -> None:
        prompt = load_annotation_prompt()

        self.assertIn("不受固定词表约束", prompt)
        self.assertNotIn("受控 dependency topic 目录", prompt)


if __name__ == "__main__":
    unittest.main()
