from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from rag.indexing.dependency_graph import (
    DocumentDependencyAggregator,
    build_dependency_notes,
    iter_note_batches,
)
from rag.indexing.semantic_annotator import normalize_annotation


def _node(node_id: str, text: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(id_=node_id, text=text, metadata=metadata)


class AnnotationNoteTests(unittest.TestCase):
    def test_keeps_grounded_rule_for_config_scope(self) -> None:
        source = "生产环境的全部回调配置都必须填写签名密钥"
        annotation = normalize_annotation(
            {
                "summary": "回调配置必须填写签名密钥",
                "subject": {"kind": "规则", "title": "回调配置要求", "identifiers": ["不存在的键"]},
                "stated_rules": [
                    {
                        "requirement": "填写签名密钥",
                        "applies_to": "生产环境的全部回调配置",
                        "required": True,
                        "identifiers": [],
                        "evidence": source,
                    }
                ],
            },
            source_text=source,
        )

        self.assertEqual(annotation["subject"]["kind"], "规则")
        self.assertEqual(annotation["subject"]["identifiers"], [])
        self.assertEqual(annotation["stated_rules"][0]["applies_to"], "生产环境的全部回调配置")
        self.assertTrue(annotation["stated_rules"][0]["required"])

    def test_keeps_paraphrased_signature_rule(self) -> None:
        source = (
            "所有接口均需要做签名校验，校验方法为设置HTTP请求头X-Signature为签名值，"
            "签名值通过**Base64URLSafe(HMAC\\_SHA\\_256 (signatureKey))**生成。"
        )
        annotation = normalize_annotation(
            {
                "subject": {"kind": "规则", "title": "验签说明", "identifiers": ["X-Signature"]},
                "stated_rules": [
                    {
                        "requirement": "做签名校验",
                        "applies_to": "全部接口",
                        "required": "true",
                        "identifiers": ["X-Signature", "HMAC_SHA_256"],
                        "evidence": "所有接口需用 X-Signature 做 HMAC_SHA_256 签名校验",
                    }
                ],
            },
            source_text=source,
        )

        rule = annotation["stated_rules"][0]
        self.assertEqual(rule["requirement"], "做签名校验")
        self.assertEqual(rule["applies_to"], "全部接口")
        self.assertTrue(rule["required"])
        self.assertEqual(rule["identifiers"], ["X-Signature", "HMAC_SHA_256"])
        self.assertIn("签名校验", rule["evidence"])

    def test_drops_rule_missing_scope_or_evidence(self) -> None:
        annotation = normalize_annotation(
            {
                "stated_rules": [
                    {
                        "requirement": "做签名校验",
                        "applies_to": "",
                        "required": True,
                        "identifiers": [],
                        "evidence": "所有接口均需要做签名校验",
                    }
                ]
            },
            source_text="所有接口均需要做签名校验。",
        )

        self.assertEqual(annotation["stated_rules"], [])


class DependencyNoteTests(unittest.TestCase):
    def test_notes_omit_full_text(self) -> None:
        node = _node(
            "api-1",
            "这里是一段很长的接口正文",
            {
                "chunk_index": 2,
                "heading_path": "接口 > 生成订单",
                "subject": {"kind": "接口", "title": "生成订单", "identifiers": ["/pay/v1/order/create"]},
                "stated_rules": [],
            },
        )

        notes = build_dependency_notes([node])

        self.assertEqual(notes[0]["node_id"], "api-1")
        self.assertNotIn("text", notes[0])
        self.assertEqual(notes[0]["subject"]["identifiers"], ["/pay/v1/order/create"])

    def test_long_document_stays_one_request(self) -> None:
        notes = [
            {
                "node_id": f"n{index}",
                "heading_path": f"章节{index}",
                "subject": {"kind": "配置", "title": f"配置{index}", "identifiers": []},
                "stated_rules": [],
            }
            for index in range(40)
        ]
        notes[0]["stated_rules"] = [
            {"requirement": "填写签名密钥", "applies_to": "全部回调配置", "required": True, "identifiers": [], "evidence": "证据一"},
            {"requirement": "加解密", "applies_to": "全部支付接口", "required": True, "identifiers": [], "evidence": "证据二"},
        ]

        batches = iter_note_batches(notes)

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]["chunks"]), 40)
        self.assertEqual(len(batches[0]["chunks"][0]["stated_rules"]), 2)
        self.assertNotIn("rule", batches[0])

    def test_short_document_stays_one_request(self) -> None:
        notes = [
            {
                "node_id": "rule-1",
                "heading_path": "加密说明",
                "subject": {"kind": "规则", "title": "加密说明", "identifiers": ["/pay/v1/"]},
                "stated_rules": [
                    {
                        "requirement": "加解密",
                        "applies_to": "所有 /pay/v1/ 路径下的接口",
                        "required": True,
                        "identifiers": ["/pay/v1/"],
                        "evidence": "所有 /pay/v1/ 路径下的接口强制要求加解密",
                    }
                ],
            },
            {
                "node_id": "api-1",
                "heading_path": "生成订单",
                "subject": {"kind": "接口", "title": "生成订单", "identifiers": ["/pay/v1/order/create"]},
                "stated_rules": [],
            },
        ]

        self.assertEqual(len(iter_note_batches(notes)), 1)


class _Completions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DependencyOrganizeTests(unittest.TestCase):
    def test_document_without_rules_does_not_call_model(self) -> None:
        completions = _Completions(["{}"])
        aggregator = DocumentDependencyAggregator(api_key="", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), prompt="整理笔记")
        node = _node("api-1", "只有参数表", {"chunk_index": 0, "markdown_relative_path": "pay.md", "subject": {"kind": "接口", "title": "生成订单", "identifiers": []}, "stated_rules": []})

        result = aggregator.aggregate_documents([node], {"pay.md": "hash"}, ".", show_progress=False)

        self.assertEqual(completions.calls, [])
        self.assertEqual(result.failed_docs, [])
        self.assertEqual(result.documents["pay.md"]["node_dependencies"]["api-1"], [])

    def test_model_receives_notes_and_program_fills_empty_nodes(self) -> None:
        edge = {
            "node_id": "rule-1",
            "topic": "加解密",
            "relation_type": "prerequisite",
            "required": True,
            "confidence": 0.95,
            "reason": "所有 /pay/v1/ 路径下的接口强制要求加解密",
        }
        completions = _Completions([json.dumps({"node_dependencies": {"api-1": [edge]}}, ensure_ascii=False)])
        aggregator = DocumentDependencyAggregator(api_key="", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), prompt="整理笔记")
        rule = _node(
            "rule-1",
            "所有 /pay/v1/ 路径下的接口强制要求加解密",
            {
                "chunk_index": 0,
                "markdown_relative_path": "pay.md",
                "heading_path": "加密说明",
                "subject": {"kind": "规则", "title": "加密说明", "identifiers": ["/pay/v1/"]},
                "stated_rules": [
                    {
                        "requirement": "加解密",
                        "applies_to": "所有 /pay/v1/ 路径下的接口",
                        "required": True,
                        "identifiers": ["/pay/v1/"],
                        "evidence": "所有 /pay/v1/ 路径下的接口强制要求加解密",
                    }
                ],
            },
        )
        api = _node(
            "api-1",
            "金额、订单号",
            {
                "chunk_index": 1,
                "markdown_relative_path": "pay.md",
                "heading_path": "生成订单",
                "subject": {"kind": "接口", "title": "生成订单", "identifiers": ["/pay/v1/order/create"]},
                "stated_rules": [],
            },
        )

        result = aggregator.aggregate_documents([rule, api], {"pay.md": "hash"}, ".", show_progress=False)

        serialized = str(completions.calls[0]["messages"][1]["content"])
        payload = json.loads(serialized)
        self.assertNotIn("text", payload["chunks"][1])
        self.assertNotIn("金额、订单号", serialized)
        self.assertEqual(result.documents["pay.md"]["node_dependencies"]["api-1"][0]["node_id"], "rule-1")
        self.assertEqual(result.documents["pay.md"]["node_dependencies"]["rule-1"], [])
