from __future__ import annotations

import unittest

from rag.agent.session import ANSWER_HISTORY_CHARS, CHAT, FOLLOW_UP, KNOWLEDGE, Session, normalize_route


class SessionTests(unittest.TestCase):
    def test_chat_does_not_retrieve(self) -> None:
        session = Session()
        prepared = session.prepare("你好")

        self.assertEqual(prepared.route, CHAT)
        self.assertFalse(prepared.retrieve)

    def test_meta_question_does_not_retrieve(self) -> None:
        prepared = Session().prepare("你能做什么？")

        self.assertEqual(prepared.route, CHAT)
        self.assertFalse(prepared.retrieve)

    def test_follow_up_uses_previous_product_and_interface(self) -> None:
        session = Session()
        first = session.prepare("谷歌订阅如何接入 linkedPurchaseToken")
        session.record(first)

        follow_up = session.prepare("它的回调验签呢")

        self.assertEqual(first.route, KNOWLEDGE)
        self.assertEqual(follow_up.route, FOLLOW_UP)
        self.assertTrue(follow_up.retrieve)
        self.assertEqual(follow_up.resolved_question, "谷歌订阅 linkedPurchaseToken 它的回调验签呢")

    def test_chat_between_turns_keeps_topic(self) -> None:
        session = Session()
        session.record(session.prepare("谷歌订阅如何接入"))
        session.record(session.prepare("谢谢"))

        follow_up = session.prepare("继续说回调")

        self.assertEqual(follow_up.route, FOLLOW_UP)
        self.assertIn("谷歌订阅", follow_up.resolved_question)

    def test_new_product_replaces_previous_topic(self) -> None:
        session = Session()
        session.record(session.prepare("谷歌订阅如何接入"))

        switched = session.prepare("支付宝周期扣款怎么接入")
        session.record(switched)
        follow_up = session.prepare("它的回调呢")

        self.assertEqual(switched.route, KNOWLEDGE)
        self.assertEqual(follow_up.resolved_question, "支付宝 它的回调呢")
        self.assertNotIn("谷歌订阅", follow_up.resolved_question)

    def test_follow_up_without_history_stays_knowledge_question(self) -> None:
        prepared = Session().prepare("它的回调验签呢")

        self.assertEqual(prepared.route, KNOWLEDGE)
        self.assertEqual(prepared.resolved_question, "它的回调验签呢")

    def test_custom_rewriter_receives_prior_questions(self) -> None:
        seen: dict[str, object] = {}

        def rewrite(questions: list[str], question: str) -> str:
            seen["questions"] = list(questions)
            seen["current"] = question
            return "谷歌订阅的超时时间"

        session = Session(rewriter=rewrite)
        session.record(session.prepare("谷歌订阅如何接入"), "先配置服务账号。")
        session.record(session.prepare("回调地址怎么填"), "填到控制台。")

        follow_up = session.prepare("超时呢")

        self.assertEqual(seen["questions"], ["谷歌订阅如何接入", "回调地址怎么填"])
        self.assertEqual(seen["current"], "超时呢")
        self.assertEqual(follow_up.resolved_question, "谷歌订阅的超时时间")

    def test_answer_history_keeps_three_turns_and_truncates_answers(self) -> None:
        session = Session()
        for index in range(4):
            prepared = session.prepare(f"谷歌订阅问题{index} linkedPurchaseToken")
            session.record(prepared, f"第{index}轮。" + ("答" * 500))

        messages = session.answer_messages()
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"] * 3)
        self.assertIn("问题1", messages[0]["content"])
        self.assertNotIn("问题0", "".join(item["content"] for item in messages))
        self.assertGreater(len(session.history[0].answer), ANSWER_HISTORY_CHARS)
        self.assertLessEqual(len(messages[1]["content"]), ANSWER_HISTORY_CHARS)
        self.assertTrue(messages[1]["content"].startswith("第1轮。"))
        self.assertEqual(session.prior_questions()[0], "谷歌订阅问题0 linkedPurchaseToken")
        self.assertEqual(len(session.prior_questions()), 4)

    def test_current_tool_context_stays_after_history(self) -> None:
        from rag.agent.tools import assemble_tool_messages

        messages = assemble_tool_messages(
            [
                {"role": "user", "content": "谷歌订阅怎么接入"},
                {"role": "assistant", "content": "先配置服务账号。"},
            ],
            "谷歌订阅服务账号的密钥如何配置",
        )
        messages.append({"role": "assistant", "content": ""})
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": "密钥片段"})

        self.assertEqual(messages[1]["content"], "谷歌订阅怎么接入")
        self.assertEqual(messages[-3]["role"], "user")
        self.assertIn("密钥如何配置", str(messages[-3]["content"]))
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertNotIn("tool", [item["role"] for item in messages[:-2]])
        self.assertIn("最近几轮对话", str(messages[0]["content"]))

    def test_router_overrides_fixed_words_and_only_follow_up_is_rewritten(self) -> None:
        rewritten: list[str] = []

        def rewrite(questions: list[str], question: str) -> str:
            rewritten.append(question)
            self.assertEqual(questions, ["你好，谷歌订阅如何接入"])
            return "谷歌订阅的超时时间"

        def router(questions: list[str], current: str) -> str:
            if "超时" in current:
                return '{"route":"follow_up"}'
            if current == "谢谢":
                return "chat"
            return "knowledge"

        session = Session(rewriter=rewrite, router=router)
        first = session.prepare("你好，谷歌订阅如何接入")
        session.record(first, "先配置服务账号。")
        thanks = session.prepare("谢谢")
        follow_up = session.prepare("超时呢")

        self.assertEqual(first.route, KNOWLEDGE)
        self.assertEqual(first.resolved_question, "你好，谷歌订阅如何接入")
        self.assertEqual(thanks.route, CHAT)
        self.assertFalse(thanks.retrieve)
        self.assertEqual(follow_up.route, FOLLOW_UP)
        self.assertEqual(follow_up.resolved_question, "谷歌订阅的超时时间")
        self.assertEqual(rewritten, ["超时呢"])

    def test_router_failure_falls_back_to_rules(self) -> None:
        def router(questions: list[str], current: str) -> str:
            raise RuntimeError("路由不可用")

        prepared = Session(router=router).prepare("你好")

        self.assertEqual(prepared.route, CHAT)
        self.assertFalse(prepared.retrieve)

    def test_follow_up_without_history_is_not_rewritten(self) -> None:
        prepared = Session(router=lambda questions, current: "follow_up").prepare("它的回调呢")

        self.assertEqual(prepared.route, KNOWLEDGE)
        self.assertEqual(prepared.resolved_question, "它的回调呢")

    def test_normalize_route_reads_json_object(self) -> None:
        self.assertEqual(normalize_route('说明\n{"route":"knowledge"}'), KNOWLEDGE)
        self.assertEqual(normalize_route("follow_up"), FOLLOW_UP)
        self.assertEqual(normalize_route("不确定"), "")

    def test_chat_does_not_enter_answer_history(self) -> None:
        session = Session()
        session.record(session.prepare("谷歌订阅如何接入"), "先配置服务账号。")
        session.record(session.prepare("谢谢"), "不客气")

        messages = session.answer_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "先配置服务账号。")


if __name__ == "__main__":
    unittest.main()
