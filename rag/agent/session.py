"""多轮会话状态、问题路由和追问补全。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

CHAT = "chat"
FOLLOW_UP = "follow_up"
KNOWLEDGE = "knowledge"

DOMAIN_TERMS = (
    "谷歌订阅",
    "google订阅",
    "apple订阅",
    "苹果订阅",
    "ios订阅",
    "支付宝",
    "微信支付",
    "paypal订阅",
    "paypal",
    "华为订阅",
    "huawei",
    "三星订阅",
    "samsung",
    "支付网关",
    "go订阅",
)
CHAT_PHRASES = (
    "你好",
    "您好",
    "在吗",
    "hi",
    "hello",
    "谢谢",
    "多谢",
    "你是谁",
    "你能做什么",
    "你会什么",
    "怎么用你",
    "怎么使用你",
)
FOLLOW_UP_TERMS = (
    "它",
    "这个",
    "那个",
    "上述",
    "上面",
    "刚才",
    "前面",
    "继续",
    "还有呢",
    "然后呢",
    "具体呢",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.]{2,}")
MAX_ANCHORS = 6
# 会话里保留完整回答。放进回答模型上下文时才截断，避免三轮历史把上下文撑满。
ANSWER_HISTORY_TURNS = 3
ANSWER_HISTORY_CHARS = 400
# 改写只用用户提问，并且可以看比回答窗口更早的几轮，用来补产品名和接口名。
REWRITE_QUESTION_TURNS = 8

FollowUpRewriter = Callable[[list[str], str], str]


@dataclass(frozen=True)
class DialogueTurn:
    """一轮已完成的知识问答。回答按原文保存。"""

    question: str
    answer: str


@dataclass(frozen=True)
class PreparedQuestion:
    """一轮问题经过路由和补全后的结果。"""

    route: str
    question: str
    resolved_question: str
    retrieve: bool


@dataclass
class Session:
    """保存最近主题和三轮问答，并把追问交给改写器补成独立问题。"""

    rewriter: FollowUpRewriter | None = None
    anchors: list[str] = field(default_factory=list)
    turns: list[PreparedQuestion] = field(default_factory=list)
    history: list[DialogueTurn] = field(default_factory=list)
    question_log: list[str] = field(default_factory=list)

    def prepare(self, question: str) -> PreparedQuestion:
        """判断问题类型。追问会用历史提问改写成独立问题。"""
        text = question.strip()
        route = self.route(text)
        resolved = text
        if route == FOLLOW_UP:
            resolved = self._complete(text)
        return PreparedQuestion(
            route=route,
            question=text,
            resolved_question=resolved,
            retrieve=route != CHAT,
        )

    def record(self, prepared: PreparedQuestion, answer: str = "") -> None:
        """记录一轮知识问题或追问，并更新后续补全使用的主题。"""
        self.turns.append(prepared)
        if prepared.route == CHAT:
            return
        self.anchors = update_anchors(self.anchors, prepared.resolved_question)
        self.question_log.append(prepared.question)
        del self.question_log[:-REWRITE_QUESTION_TURNS]
        self.history.append(DialogueTurn(prepared.question, answer.strip()))
        del self.history[:-ANSWER_HISTORY_TURNS]

    def prior_questions(self) -> list[str]:
        """返回改写可用的历史提问，比回答窗口更长。"""
        return list(self.question_log)

    def answer_messages(self) -> list[dict[str, str]]:
        """返回最近三轮问答，供回答模型理解指代。不含工具调用记录。"""
        messages: list[dict[str, str]] = []
        for turn in self.history:
            messages.append({"role": "user", "content": turn.question})
            if turn.answer:
                messages.append({"role": "assistant", "content": truncate_answer(turn.answer)})
        return messages

    def route(self, question: str) -> str:
        """闲聊直接回答，带指代的追问先补全，其余问题进入检索。"""
        if _is_chat(question) and not extract_domain_terms(question):
            return CHAT
        if self.anchors and _is_follow_up(question) and not _switches_topic(question, self.anchors):
            return FOLLOW_UP
        return KNOWLEDGE

    def _complete(self, question: str) -> str:
        if self.rewriter is not None:
            rewritten = self.rewriter(self.prior_questions(), question).strip()
            if rewritten:
                return rewritten
        return complete_follow_up(question, self.anchors)


def truncate_answer(answer: str, limit: int = ANSWER_HISTORY_CHARS) -> str:
    """放进模型上下文时把回答截到段落长度，并尽量停在句号或换行处。"""
    text = answer.strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    for separator in ("。", "！", "？", "\n"):
        index = window.rfind(separator)
        if index >= limit // 2:
            return window[: index + 1]
    return window[: limit - 1].rstrip() + "…"


def complete_follow_up(question: str, anchors: list[str]) -> str:
    """把历史主题中当前问题没有出现的词补到问题前面。"""
    missing = [anchor for anchor in anchors if anchor and anchor not in question]
    if not missing:
        return question
    return f"{' '.join(missing)} {question}".strip()


def update_anchors(anchors: list[str], question: str) -> list[str]:
    """用本轮问题更新主题。出现新产品时替换旧的产品主题。"""
    products = extract_domain_terms(question)
    tokens = extract_tokens(question)
    if products:
        return _dedupe(products + tokens)[:MAX_ANCHORS]
    return _dedupe(anchors + tokens)[:MAX_ANCHORS]


def extract_domain_terms(question: str) -> list[str]:
    """按原文大小写提取问题里的产品或业务名。"""
    folded = question.casefold()
    found: list[str] = []
    for term in DOMAIN_TERMS:
        index = folded.find(term.casefold())
        if index >= 0:
            found.append(question[index:index + len(term)])
    return _dedupe(found)


def extract_tokens(question: str) -> list[str]:
    """提取接口名、错误码这类连续的英文技术词。"""
    products = {term.casefold() for term in extract_domain_terms(question)}
    tokens: list[str] = []
    for match in TOKEN_PATTERN.findall(question):
        if match.casefold() in products:
            continue
        tokens.append(match)
    return _dedupe(tokens)


def _is_chat(question: str) -> bool:
    compact = re.sub(r"[\s？?！!。.~，,]", "", question).casefold()
    if compact in {phrase.casefold() for phrase in CHAT_PHRASES}:
        return True
    return any(phrase.casefold() in compact for phrase in ("你能做什么", "你是谁", "你会什么"))


def _is_follow_up(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    if any(term in compact for term in FOLLOW_UP_TERMS):
        return True
    return compact.endswith("呢") and len(compact) <= 24


def _switches_topic(question: str, anchors: list[str]) -> bool:
    anchor_keys = {anchor.casefold() for anchor in anchors}
    return any(term.casefold() not in anchor_keys for term in extract_domain_terms(question))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result
