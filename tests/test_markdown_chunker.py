from __future__ import annotations

import unittest

from rag.indexing.markdown_chunker import H3_ATOMIC_MAX_TOKENS, split_markdown_chunks
from rag.retrieval.tokenization import estimate_token_size


def _repeat_until(text: str, token_target: int) -> str:
    parts: list[str] = []
    while estimate_token_size("\n\n".join(parts)) < token_target:
        parts.append(text)
    return "\n\n".join(parts)


class MarkdownChunkerTests(unittest.TestCase):
    def test_oversized_atomic_section_is_kept(self) -> None:
        body = _repeat_until("订单字段说明必须完整保留。", H3_ATOMIC_MAX_TOKENS + 200)
        markdown = f"# 支付接入\n\n## 接口信息\n\n### 查询订单接口\n\n{body}\n"
        self.assertGreater(estimate_token_size(body), H3_ATOMIC_MAX_TOKENS)

        chunks = split_markdown_chunks(markdown, chunk_size=512, chunk_overlap=50)
        joined = "\n".join(chunk["text"] for chunk in chunks)
        sentence = "订单字段说明必须完整保留。"

        # overlap 会让相邻 chunk 重复少量句子，所以这里只确认正文没有被整节丢掉。
        self.assertGreaterEqual(joined.count(sentence), markdown.count(sentence))
        self.assertTrue(any("查询订单接口" in chunk["heading_path"] for chunk in chunks))
        self.assertFalse(any(_is_heading_only(chunk["text"]) for chunk in chunks))

    def test_sibling_headings_do_not_nest_and_h1_is_kept(self) -> None:
        markdown = """# 支付接入

## 接口信息

### 生成预支付订单

创建订单。

### 查询用户订单列表

查询订单。
"""
        chunks = split_markdown_chunks(markdown, chunk_size=512, chunk_overlap=50)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["heading_path"].startswith("支付接入") for chunk in chunks))
        for chunk in chunks:
            self.assertNotIn("生成预支付订单 > 查询用户订单列表", chunk["heading_path"])
            self.assertNotIn("查询用户订单列表 > 生成预支付订单", chunk["heading_path"])
        covered = next(chunk for chunk in chunks if "生成预支付订单" in chunk["text"] and "查询用户订单列表" in chunk["text"])
        self.assertEqual(covered["heading_path"], "支付接入 > 接口信息")

    def test_separate_sibling_sections_keep_their_own_path(self) -> None:
        first = _repeat_until("邮件申请需要提交产品名称。", 400)
        second = _repeat_until("实时通知需要配置回调地址。", 400)
        markdown = f"""# Apple订阅服务接入

## 通知

### 1、邮件申请

{first}

### 2、实时通知

{second}
"""
        chunks = split_markdown_chunks(markdown, chunk_size=512, chunk_overlap=40)
        paths = [chunk["heading_path"] for chunk in chunks]

        self.assertTrue(any(path.endswith("1、邮件申请") for path in paths))
        self.assertTrue(any(path.endswith("2、实时通知") for path in paths))
        self.assertTrue(all(path.startswith("Apple订阅服务接入") for path in paths))
        self.assertFalse(any("邮件申请 > " in path for path in paths))

    def test_heading_only_fragment_merges_into_neighbor(self) -> None:
        markdown = """# 支付接入和权限申请指引

## 接口

### 内购订单支付接口

## 调用说明

这里说明如何发起内购支付。
"""
        chunks = split_markdown_chunks(markdown, chunk_size=512, chunk_overlap=50)

        self.assertEqual(len(chunks), 1)
        self.assertIn("支付接入和权限申请指引", chunks[0]["text"])
        self.assertIn("内购订单支付接口", chunks[0]["text"])
        self.assertIn("这里说明如何发起内购支付。", chunks[0]["text"])
        self.assertFalse(_is_heading_only(chunks[0]["text"]))
        self.assertGreaterEqual(len(chunks[0]["text"]), 40)


def _is_heading_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("#") for line in lines)


if __name__ == "__main__":
    unittest.main()
