from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.indexing.storage_indexer import StoredBM25Index, tokens_for_bm25
from rag.retrieval.retriever import BM25KeywordRetriever
from rag.retrieval.tokenization import tokenize_for_bm25, tokenize_technical_text


class _FixedScores:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        del query_tokens
        return list(self.scores)


class Bm25TokenizationTests(unittest.TestCase):
    def test_bm25_tokenizer_keeps_frequency_and_drops_noise(self) -> None:
        tokens = tokenize_for_bm25("谷歌订阅的回调通知怎么配置？谷歌订阅（重要）什么")
        deduped = tokenize_technical_text("signature signature signature")

        self.assertGreaterEqual(tokens.count("谷歌订阅"), 2)
        self.assertNotIn("的", tokens)
        self.assertNotIn("怎么", tokens)
        self.assertNotIn("什么", tokens)
        self.assertFalse(any(token in {"？", "（", "）", "?"} for token in tokens))
        self.assertEqual(tokenize_for_bm25("signature signature signature").count("signature"), 3)
        self.assertEqual(deduped.count("signature"), 1)
        self.assertGreaterEqual(tokenize_for_bm25("通知。通知。").count("通知"), 2)

    def test_index_tokens_keep_repeated_terms(self) -> None:
        tokens = tokens_for_bm25(
            {"text": "签名校验。签名校验。", "summary": "", "metadata": {}},
            tokenize_for_bm25,
        )

        self.assertGreaterEqual(tokens.count("签名"), 2)

    def test_retriever_drops_non_positive_scores(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bm25.pkl"
            payload = StoredBM25Index(
                node_ids=["zero", "negative", "positive"],
                tokenized_corpus=[[], [], ["签名"]],
                bm25=_FixedScores([0.0, -0.4, 1.2]),
            )
            with path.open("wb") as file:
                pickle.dump(payload, file)

            hits = BM25KeywordRetriever(path).retrieve("签名怎么配置？", top_k=5)

        self.assertEqual([hit.node_id for hit in hits], ["positive"])
        self.assertEqual(hits[0].rank, 1)
        self.assertGreater(hits[0].raw_score, 0)


if __name__ == "__main__":
    unittest.main()
