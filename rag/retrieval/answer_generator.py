from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from rag.retrieval.retriever import RetrievalCandidate


@dataclass(frozen=True)
class AnswerReference:
    """答案使用到的引用来源。"""

    node_id: str
    file_name: str
    chunk_index: str
    score: float


@dataclass(frozen=True)
class GeneratedAnswer:
    """答案生成结果。"""

    answer: str
    references: list[AnswerReference]


class AnswerGenerator:
    """基于检索结果生成最终答案。"""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "qwen3.6-plus",
        enabled: bool = True,
        context_top_k: int = 4,
        max_context_chars: int = 1200,
    ) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.context_top_k = context_top_k
        self.max_context_chars = max_context_chars

    def generate(self, question: str, candidates: list[RetrievalCandidate]) -> GeneratedAnswer:
        references = [
            AnswerReference(
                node_id=candidate.node_id,
                file_name=str(candidate.metadata.get("file_name") or candidate.metadata.get("doc_id") or "unknown"),
                chunk_index=str(candidate.metadata.get("chunk_index", "?")),
                score=candidate.final_score,
            )
            for candidate in candidates[: self.context_top_k]
        ]
        if not references:
            return GeneratedAnswer(answer="根据现有文档无法回答。", references=[])

        if not self.enabled or self.client is None:
            return self._build_fallback_answer(candidates[: self.context_top_k], references)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是企业知识库问答助手。"
                        "你只能依据提供的参考资料回答，不允许补充资料之外的事实。"
                        "如果资料不足，请明确回答“根据现有文档无法回答”。"
                        "请优先给出步骤、结论、配置项或注意事项，并在答案末尾追加“参考来源：”行。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(question, candidates[: self.context_top_k]),
                },
            ],
            temperature=0.3,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return self._build_fallback_answer(candidates[: self.context_top_k], references)
        return GeneratedAnswer(answer=answer, references=references)

    def _build_user_prompt(self, question: str, candidates: list[RetrievalCandidate]) -> str:
        contexts: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            file_name = str(candidate.metadata.get("file_name") or candidate.metadata.get("doc_id") or "unknown")
            chunk_index = candidate.metadata.get("chunk_index", "?")
            text = candidate.text.strip()[: self.max_context_chars]
            contexts.append(
                f"[来源 {index}] file={file_name} | chunk={chunk_index} | node_id={candidate.node_id}\n{text}"
            )
        payload = {
            "question": question,
            "contexts": contexts,
            "answer_requirements": [
                "基于参考资料回答，不要超出资料范围",
                "优先输出可执行步骤或明确结论",
                "资料不足时明确说明无法回答",
                "最后一行输出参考来源，格式为：参考来源：文件名 chunk-x(node_id)",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_fallback_answer(
        self,
        candidates: list[RetrievalCandidate],
        references: list[AnswerReference],
    ) -> GeneratedAnswer:
        lines = ["当前未启用答案生成模型，以下是最相关资料摘要："]
        for index, candidate in enumerate(candidates, start=1):
            file_name = str(candidate.metadata.get("file_name") or candidate.metadata.get("doc_id") or "unknown")
            chunk_index = candidate.metadata.get("chunk_index", "?")
            summary = str(candidate.metadata.get("summary") or "").strip()
            excerpt = candidate.text.strip().replace("\n", " ")[:160]
            brief = summary or excerpt or "该片段无可用摘要。"
            lines.append(
                f"{index}. {file_name} chunk-{chunk_index} ({candidate.node_id})：{brief}"
            )
        lines.append(
            "参考来源：" + "；".join(
                f"{item.file_name} chunk-{item.chunk_index}({item.node_id})"
                for item in references
            )
        )
        return GeneratedAnswer(answer="\n".join(lines), references=references)
