from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.indexing.markdown_chunker import split_markdown_chunks
from rag.preprocess.enriched_corpus import export_chunk_records, export_enriched_markdown
from rag.preprocess.image_captions import (
    extract_chunk_images,
    inject_image_captions,
    load_image_captions,
)


APPLE_IMAGE = "cleaned_assets/支付订阅服务/订阅服务(Apple)/demo/img_0001.png"


class ImageCaptionTests(unittest.TestCase):
    def test_inject_keeps_caption_on_the_same_line_for_paths_with_parentheses(self) -> None:
        markdown = f"# 接入\n\n## 回调\n\n先配置地址。\n\n![]({APPLE_IMAGE})\n\n然后保存。\n"
        enriched, missing = inject_image_captions(
            markdown,
            {APPLE_IMAGE: "图中是 Apple 后台的通知地址页，回调 URL 填的是网关地址。"},
        )

        self.assertEqual(missing, [])
        self.assertIn(f"![]({APPLE_IMAGE}) 【图片说明｜img_0001.png】图中是 Apple 后台的通知地址页", enriched)
        chunks = split_markdown_chunks(enriched, chunk_size=512, chunk_overlap=100)
        self.assertTrue(any(APPLE_IMAGE in item["text"] and "回调 URL" in item["text"] for item in chunks))

    def test_inject_is_idempotent_and_reports_missing_caption(self) -> None:
        markdown = f"![]({APPLE_IMAGE}) 【图片说明｜img_0001.png】已有说明\n\n![](cleaned_assets/missing.png)\n"
        enriched, missing = inject_image_captions(markdown, {APPLE_IMAGE: "新说明"})

        self.assertEqual(enriched.count("【图片说明｜img_0001.png】"), 1)
        self.assertIn("已有说明", enriched)
        self.assertNotIn("新说明", enriched)
        self.assertEqual(missing, ["cleaned_assets/missing.png"])

    def test_extract_chunk_images_reads_adjacent_caption(self) -> None:
        text = f"![]({APPLE_IMAGE}) 【图片说明｜img_0001.png】图中展示通知开关。\n"
        images = extract_chunk_images(text)

        self.assertEqual(images[0]["path"], APPLE_IMAGE)
        self.assertEqual(images[0]["file_name"], "img_0001.png")
        self.assertEqual(images[0]["description"], "图中展示通知开关。")

    def test_load_captions_accepts_description_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image_captions.json"
            path.write_text(
                json.dumps({"images": {APPLE_IMAGE: {"description": "  图中是配置页  "}}}, ensure_ascii=False),
                encoding="utf-8",
            )
            captions = load_image_captions(path)

        self.assertEqual(captions[APPLE_IMAGE], "图中是配置页")

    def test_export_writes_enriched_markdown_and_chunk_image_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_dir = root / "cleaned_markdown"
            relative = "支付/接入说明.md"
            source = markdown_dir / relative
            source.parent.mkdir(parents=True)
            source.write_text(f"# 接入\n\n## 流程\n\n![]({APPLE_IMAGE})\n", encoding="utf-8")
            manifest = {
                "documents": [
                    {
                        "source_path": "D:/raw/接入说明.doc",
                        "source_relative_path": "支付/接入说明.doc",
                        "source_file_name": "接入说明.doc",
                        "markdown_path": str(source),
                        "markdown_relative_path": relative,
                    }
                ]
            }
            (markdown_dir / "source_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )

            stats = export_enriched_markdown(
                markdown_dir,
                root / "enriched_markdown",
                {APPLE_IMAGE: "图中是接入流程，共三步。"},
            )
            records = export_chunk_records(root / "enriched_markdown")

        self.assertEqual(stats["captioned_image_count"], 1)
        self.assertEqual(stats["missing_caption_count"], 0)
        image_records = [record for record in records if record["images"]]
        self.assertEqual(len(image_records), 1)
        self.assertEqual(image_records[0]["images"][0]["description"], "图中是接入流程，共三步。")
        self.assertIn("图中是接入流程", image_records[0]["text"])
        self.assertEqual(image_records[0]["markdown_relative_path"], relative)
        self.assertEqual(image_records[0]["doc_id"], relative)
        self.assertEqual(image_records[0]["archive_path"], "支付")
        self.assertNotIn("source_path", image_records[0])
        self.assertNotIn("source_relative_path", image_records[0])


if __name__ == "__main__":
    unittest.main()
