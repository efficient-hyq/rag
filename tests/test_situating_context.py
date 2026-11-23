from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rag.indexing.chunk_context import build_context_text, build_retrieval_text
from rag.indexing.semantic_annotator import SemanticAnnotator
from rag.indexing.situating_context import (
    SituatingContextGenerator,
    clean_situating_context,
    load_situating_prompt,
    load_whole_documents,
    render_situating_prompt,
)
from rag.shared.checkpoints import CheckpointStore


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="```\n这一片段位于 Google 订阅说明的取消章节。\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class SituatingContextTests(unittest.TestCase):
    def test_retrieval_text_prepends_context_without_changing_annotation_text(self) -> None:
        metadata = {
            "archive_path": "支付订阅服务/Google订阅",
            "file_name": "说明.md",
            "heading_path": "取消",
            "situating_context": "这一片段位于 Google 订阅说明的取消章节。",
        }
        archive_text = build_context_text("开通后可取消。", metadata)
        retrieval_text = build_retrieval_text("开通后可取消。", metadata)

        self.assertNotIn("取消章节", archive_text)
        self.assertTrue(retrieval_text.startswith("这一片段位于 Google 订阅说明的取消章节。"))
        self.assertIn("归档：支付订阅服务/Google订阅", retrieval_text)
        self.assertIn("正文：", retrieval_text)
        self.assertIn("开通后可取消。", retrieval_text)

        node = SimpleNamespace(node_id="node-1", text="开通后可取消。", metadata=metadata)
        annotator = SemanticAnnotator(api_key="", model="demo", client=SimpleNamespace(), prompt="标注")
        self.assertEqual(annotator._annotation_input(node), retrieval_text)

    def test_prompt_keeps_document_before_chunk(self) -> None:
        template = load_situating_prompt("prompts/chunk_context.md")
        rendered = render_situating_prompt(template, "全文：取消订阅。", "开通后可取消。")
        self.assertLess(rendered.index("全文：取消订阅。"), rendered.index("开通后可取消。"))
        self.assertNotIn("{{WHOLE_DOCUMENT}}", rendered)
        self.assertNotIn("{{CHUNK_CONTENT}}", rendered)

    def test_generator_caches_context_and_keeps_original_text(self) -> None:
        completions = _Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        prompt = load_situating_prompt("prompts/chunk_context.md")
        generator = SituatingContextGenerator(
            api_key="",
            base_url="http://example",
            model="demo-model",
            prompt=prompt,
            prompt_hash="prompt-hash",
            client=client,
        )
        node = SimpleNamespace(
            node_id="node-1",
            text="开通后可取消。",
            metadata={
                "markdown_relative_path": "支付订阅服务/Google订阅/说明.md",
                "chunk_index": 0,
            },
        )
        documents = {"支付订阅服务/google订阅/说明.md": "全文：取消订阅。开通后可取消。"}
        hashes = {"支付订阅服务/google订阅/说明.md": "doc-hash"}
        with TemporaryDirectory() as temp:
            checkpoint = CheckpointStore(temp)
            generator.apply([node], documents, hashes, checkpoint)
            generator.apply([node], documents, hashes, checkpoint)

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(completions.calls[0]["model"], "demo-model")
        user_message = completions.calls[0]["messages"][1]["content"]
        self.assertIn("全文：取消订阅。", user_message)
        self.assertIn("开通后可取消。", user_message)
        self.assertEqual(node.text, "开通后可取消。")
        self.assertEqual(node.metadata["situating_context"], "这一片段位于 Google 订阅说明的取消章节。")

    def test_missing_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SituatingContextGenerator(
                api_key="key",
                base_url="http://example",
                model="",
                prompt=load_situating_prompt("prompts/chunk_context.md"),
                prompt_hash="prompt-hash",
            )

    def test_whole_document_prefers_markdown_file(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            relative = Path("支付订阅服务/Google订阅/说明.md")
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_text("文件中的全文。", encoding="utf-8")
            node = SimpleNamespace(
                text="片段。",
                metadata={"markdown_relative_path": relative.as_posix(), "chunk_index": 0},
            )
            documents = load_whole_documents([node], root)

        self.assertEqual(documents["支付订阅服务/google订阅/说明.md"], "文件中的全文。")

    def test_clean_context_drops_code_fence(self) -> None:
        self.assertEqual(clean_situating_context("```text\n位置说明。\n```"), "位置说明。")
