from __future__ import annotations

import unittest

from rag.retrieval.retired import exclude_retired_candidates, usable_text


class _Candidate:
    def __init__(self, node_id: str, text: str, metadata: dict[str, object]) -> None:
        self.node_id = node_id
        self.text = text
        self.metadata = metadata


class RetiredDocumentTests(unittest.TestCase):
    def test_retired_path_is_not_usable(self) -> None:
        metadata = {
            "archive_path": "支付订阅服务/支付网关-SDK接入文档（已废弃）/支付SDK接入文档(iOS)",
            "file_name": "支付SDK接入文档(iOS).md",
            "text": "初始化配置",
            "heading_path": "基本使用",
        }

        self.assertIsNone(usable_text(metadata))

    def test_discarded_directory_marker_is_not_usable(self) -> None:
        metadata = {
            "markdown_relative_path": "支付订阅服务/订阅服务(Apple)/（废弃）订阅服务(Apple)-常见问题解答/说明.md",
            "text": "常见问题",
            "heading_path": "常见问题",
        }

        self.assertIsNone(usable_text(metadata))

    def test_struck_section_is_removed_and_live_section_remains(self) -> None:
        text = "\n".join(
            [
                "### ~~查询订单~~ （已经过时，使用查询用户订单接口替代）",
                "~~资源路径~~ ~~ISO1940003~~",
                "### 商品下发",
                "资源路径 /api/v1/goods",
            ]
        )
        metadata = {"archive_path": "支付订阅服务/支付网关（国内）", "heading_path": "接口信息", "text": text}

        cleaned = usable_text(metadata)

        self.assertIsNotNone(cleaned)
        assert cleaned is not None
        self.assertIn("商品下发", cleaned)
        self.assertIn("/api/v1/goods", cleaned)
        self.assertNotIn("ISO1940003", cleaned)
        self.assertNotIn("已经过时", cleaned)

    def test_fully_struck_heading_without_live_section_is_dropped(self) -> None:
        metadata = {
            "heading_path": "接入协议 > ~~预订单接口 (无需精准匹配订单不需要调用)~~",
            "text": "### ~~预订单接口 (无需精准匹配订单不需要调用)~~\n\n~~资源路径~~ ~~/ISO1865005~~",
        }

        self.assertIsNone(usable_text(metadata))

    def test_table_strike_marker_stays(self) -> None:
        text = "| ~~#~~ | 参数名 |\n| --- | --- |\n| 1 | api_key |"
        metadata = {"heading_path": "请求参数", "text": text}

        self.assertEqual(usable_text(metadata), text)

    def test_expired_order_state_is_not_treated_as_retired(self) -> None:
        text = "交易状态取值有 4订单已过期"
        metadata = {"heading_path": "订单校验", "text": text}

        self.assertEqual(usable_text(metadata), text)

    def test_exclude_candidates_drops_retired_and_rewrites_mixed_text(self) -> None:
        retired = _Candidate(
            "old",
            "旧接入",
            {"archive_path": "支付网关-客户端包（已废弃）", "text": "旧接入", "heading_path": "接入"},
        )
        mixed = _Candidate(
            "mixed",
            "### ~~订阅购买~~\n旧流程\n### 现行接入\n使用新接口",
            {"archive_path": "支付流程", "text": "将被替换", "heading_path": "支付流程"},
        )

        kept, dropped = exclude_retired_candidates([retired, mixed])

        self.assertEqual(dropped, 1)
        self.assertEqual([item.node_id for item in kept], ["mixed"])
        self.assertIn("现行接入", kept[0].text)
        self.assertNotIn("旧流程", kept[0].text)
        self.assertEqual(kept[0].metadata["text"], kept[0].text)


if __name__ == "__main__":
    unittest.main()
