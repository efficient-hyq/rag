from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from rag.agent.session import CHAT, Session
from rag.config.common import PathConfig
from rag.config.retrieval import QueryConfig
from rag.retrieval.query_service import build_query_service
from rag.retrieval.retriever import HybridRetriever, RetrievalCandidate

SNIPPET_CHARS = 200
DEFAULT_TOP_K = 5
MAX_TOP_K = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中查找与问题相关的片段。query 使用适合检索的短查询，可以是用户原问题，也可以是拆出的产品名、接口名或错误码。每条结果只含 node_id、文件名、归档路径、heading_path、摘要和不超过 200 字的片段，不含全文。归档路径用于区分产品线，例如谷歌订阅和支付网关。没有命中时返回空列表。摘要不够回答时，再用 node_id 调用 read_chunk。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询，保留产品名、接口名、错误码等原词"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "返回条数，默认 5。摘要不够覆盖问题时再加大",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_chunk",
            "description": "按 node_id 读取片段全文。这是唯一返回全文的工具。node_id 必须来自 search_knowledge_base、get_dependencies、read_neighbors 或 grep_docs 的返回值，不要编造。仅当摘要或短片段不够回答时调用。不存在时返回 found=false。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "已返回的片段标识"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dependencies",
            "description": "查询一个片段的必要前置片段，例如接入前必须满足的配置、校验、回调或证书。先通过 search_knowledge_base 取得 node_id 再调用。每项返回依赖 node_id、topic、reason、summary 和不超过 200 字的片段，不含全文。没有依赖时 dependencies 为空数组。需要全文时再对依赖 node_id 调用 read_chunk。node_id 不存在时返回 found=false。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "已返回的片段标识"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_neighbors",
            "description": "读取同一篇文档中当前片段前后相邻的片段。只返回 summary 和不超过 200 字的片段，不含全文。需要全文时再对返回的 node_id 调用 read_chunk。node_id 不存在时返回 found=false。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "已返回的片段标识"},
                    "radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2,
                        "description": "向前后各扩展几段，默认 1",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_docs",
            "description": "在清洗后的 Markdown 中精确查找接口名、错误码、字段名或配置键。适合向量检索容易漏掉的原词。结果按文档轮流排列，同一文档内按 chunk_index 从前往后。每条结果包含 node_id、文件绝对路径、summary 和从该片段截取的不超过 200 字命中片段，不含全文。需要全文时再调用 read_chunk。超过 max_results 时截断，omitted_count 表示未返回的命中数，next_offset 是下一页起点。没有命中时 matches 为空数组，next_offset 为 null。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "要精确匹配的原文，例如接口名、错误码或字段名"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "最多返回几条，默认 8",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "从第几条命中开始返回，默认 0。继续查找时传入上次结果的 next_offset",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


def read_chunk(node_id: str) -> str:
    """按 node_id 读取已发布索引中的片段全文，返回 JSON 字符串。"""
    logger = logging.getLogger("rag.agent.tools")
    normalized_id = str(node_id or "").strip()
    if not normalized_id:
        return _dump({"found": False, "error": "node_id 不能为空"})
    try:
        metadata = _get_retriever().metadata_index.metadata_by_id.get(normalized_id)
        if metadata is None:
            logger.info("片段不存在 | node_id=%s", normalized_id)
            return _dump({"found": False, "node_id": normalized_id})
        file_path = chunk_file_path(metadata, _get_docs_dir())
    except Exception as exc:
        logger.exception("读取片段失败 | node_id=%s", normalized_id)
        return _dump({"found": False, "node_id": normalized_id, "error": str(exc)})

    file_name = str(metadata.get("file_name") or "")
    if not file_name and file_path:
        file_name = Path(file_path).name
    payload = {
        "found": True,
        "node_id": normalized_id,
        "file_name": file_name,
        "file_path": file_path,
        "heading_path": str(metadata.get("heading_path") or ""),
        "chunk_index": metadata.get("chunk_index"),
        "text": str(metadata.get("text") or ""),
    }
    logger.info(
        "读取片段完成 | node_id=%s | file_path=%s | 字数=%s",
        normalized_id,
        file_path,
        len(payload["text"]),
    )
    return _dump(payload)


def get_dependencies(node_id: str) -> str:
    """查询片段的必要前置依赖，返回 JSON 字符串。"""
    logger = logging.getLogger("rag.agent.tools")
    normalized_id = str(node_id or "").strip()
    if not normalized_id:
        return _dump({"found": False, "error": "node_id 不能为空"})
    try:
        retriever = _get_retriever()
        if normalized_id not in retriever.metadata_index.metadata_by_id:
            logger.info("片段不存在 | node_id=%s", normalized_id)
            return _dump({"found": False, "node_id": normalized_id})
        edges = retriever.dependency_graph.get_required_edges(normalized_id)
    except Exception as exc:
        logger.exception("查询依赖失败 | node_id=%s", normalized_id)
        return _dump({"found": False, "node_id": normalized_id, "error": str(exc)})

    dependencies = []
    for edge in edges:
        target_id = str(edge.get("node_id") or "").strip()
        if not target_id:
            continue
        item = _summary_view(target_id, retriever.metadata_index.metadata_by_id.get(target_id, {}))
        item["topic"] = str(edge.get("topic") or "")
        item["reason"] = str(edge.get("reason") or "")
        dependencies.append(item)
    logger.info("查询依赖完成 | node_id=%s | 依赖数=%s", normalized_id, len(dependencies))
    return _dump({"found": True, "node_id": normalized_id, "dependencies": dependencies})


def read_neighbors(node_id: str, radius: int = 1) -> str:
    """读取同一文档内的相邻片段摘要，返回 JSON 字符串。"""
    logger = logging.getLogger("rag.agent.tools")
    normalized_id = str(node_id or "").strip()
    if not normalized_id:
        return _dump({"found": False, "error": "node_id 不能为空"})
    try:
        metadata_index = _get_retriever().metadata_index
        metadata = metadata_index.metadata_by_id.get(normalized_id)
        if metadata is None:
            logger.info("片段不存在 | node_id=%s", normalized_id)
            return _dump({"found": False, "node_id": normalized_id})
        distance = _normalize_radius(radius)
        neighbors = _neighbor_views(metadata_index, normalized_id, metadata, distance)
    except Exception as exc:
        logger.exception("读取相邻片段失败 | node_id=%s", normalized_id)
        return _dump({"found": False, "node_id": normalized_id, "error": str(exc)})
    logger.info("读取相邻片段完成 | node_id=%s | radius=%s | 数量=%s", normalized_id, distance, len(neighbors))
    return _dump({"found": True, "node_id": normalized_id, "neighbors": neighbors})


def grep_docs(pattern: str, max_results: int = MAX_TOP_K, offset: int = 0) -> str:
    """在清洗后的 Markdown 中精确查找原文，返回 JSON 字符串。"""
    logger = logging.getLogger("rag.agent.tools")
    normalized = str(pattern or "").strip()
    if not normalized:
        return _dump({"error": "pattern 不能为空"})
    if len(normalized) > 80:
        return _dump({"error": "pattern 过长"})
    try:
        limit = _normalize_top_k(max_results)
        start = _normalize_offset(offset)
        docs_dir = _get_docs_dir().resolve()
        if not docs_dir.is_dir():
            return _dump({"error": f"文档目录不存在: {docs_dir}"})
        matches, omitted_count, next_offset = _grep_markdown(
            docs_dir,
            _get_retriever().metadata_index,
            normalized,
            limit,
            start,
        )
    except Exception as exc:
        logger.exception("精确查找失败 | pattern=%s", normalized)
        return _dump({"error": str(exc)})
    logger.info(
        "精确查找完成 | pattern=%s | offset=%s | 命中数=%s | 未返回=%s | 下一页=%s",
        normalized,
        start,
        len(matches),
        omitted_count,
        next_offset,
    )
    return _dump({"matches": matches, "omitted_count": omitted_count, "next_offset": next_offset})


def search_knowledge_base(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """检索知识库片段，返回 JSON 字符串。"""
    logger = logging.getLogger("rag.agent.tools")
    if not query or not str(query).strip():
        return _dump({"error": "query 不能为空"})
    try:
        limit = _normalize_top_k(top_k)
        candidates = _get_retriever().search_snippets(str(query).strip(), limit)
    except Exception as exc:
        logger.exception("知识库检索失败 | query=%s", query)
        return _dump({"error": str(exc)})
    payload = [_candidate_payload(item) for item in candidates]
    logger.info("知识库检索完成 | query=%s | top_k=%s | 返回数=%s", query, limit, len(payload))
    return _dump(payload)


@lru_cache(maxsize=1)
def _get_query_config() -> QueryConfig:
    config = QueryConfig.from_env()
    storage_dir = resolve_storage_dir(os.getenv("RAG_STORAGE_DIR"), config.paths.storage_dir)
    docs_dir = resolve_docs_dir(os.getenv("RAG_DOCS_DIR"), storage_dir, _project_docs_dir())
    return replace(config, paths=PathConfig(docs_dir=docs_dir, storage_dir=storage_dir))


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    return build_query_service(_get_query_config()).retriever


@lru_cache(maxsize=1)
def _get_docs_dir() -> Path:
    return _get_query_config().paths.docs_dir


def resolve_storage_dir(env_value: str | None, default: Path) -> Path:
    """选择索引根目录。相对路径按项目根目录解析，不随启动目录变化。"""
    from rag.config.common import resolve_project_path

    if env_value and str(env_value).strip():
        return resolve_project_path(env_value)
    return Path(default)


def resolve_docs_dir(env_value: str | None, storage_dir: Path, project_docs_dir: Path) -> Path:
    """选择预处理 Markdown 的根目录，供相对路径校验。优先带图片说明的目录。"""
    from rag.config.common import resolve_project_path

    if env_value and str(env_value).strip():
        return resolve_project_path(env_value)
    enriched = storage_dir / "enriched_markdown"
    if enriched.is_dir():
        return enriched
    cleaned = storage_dir / "cleaned_markdown"
    if cleaned.is_dir():
        return cleaned
    if project_docs_dir.is_dir():
        return project_docs_dir
    return enriched


def _project_docs_dir() -> Path:
    from rag.config.common import project_root

    return project_root() / "storage" / "enriched_markdown"


def chunk_file_path(metadata: dict[str, object], docs_dir: Path) -> str:
    """返回预处理 Markdown 文件的绝对路径。

    新索引使用 markdown_relative_path。旧索引仍可能只有 cleaned_markdown_relative_path。
    """
    root = docs_dir.resolve()
    stored = _stored_markdown_relative(metadata)
    if stored:
        candidate = (root / stored).resolve()
        if _is_under(candidate, root):
            return str(candidate)
        return ""
    resolved = _resolve_under_docs_dir(metadata, root)
    if resolved is None:
        return ""
    return str(resolved)


def _stored_markdown_relative(metadata: dict[str, object]) -> str:
    for key in ("markdown_relative_path", "cleaned_markdown_relative_path"):
        stored = _safe_relative(metadata.get(key))
        if stored:
            return stored
    return ""


def _safe_relative(value: object) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


def _resolve_under_docs_dir(metadata: dict[str, object], docs_dir: Path) -> Path | None:
    root = docs_dir.resolve()
    for key in ("cleaned_markdown_path", "file_path", "doc_id"):
        located = _locate_existing_file(metadata.get(key), root)
        if located is not None:
            return located
    return None


def _locate_existing_file(value: object, docs_root: Path) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    parts = tuple(part for part in Path(raw).parts if part not in {"", "."})
    candidates = [Path(raw)] if Path(raw).is_absolute() else [Path.cwd() / raw]
    for index in range(len(parts)):
        if ".." in parts[index:]:
            continue
        candidates.append(docs_root.joinpath(*parts[index:]))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_under(resolved, docs_root) and resolved.is_file():
            return resolved
    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalize_offset(offset: int | None) -> int:
    if offset is None:
        return 0
    return max(int(offset), 0)


def _normalize_top_k(top_k: int | None) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    return min(max(int(top_k), 1), MAX_TOP_K)


def _candidate_payload(candidate: RetrievalCandidate) -> dict[str, str]:
    metadata = dict(candidate.metadata)
    metadata.setdefault("text", candidate.text)
    if not metadata.get("file_name"):
        metadata["file_name"] = str(metadata.get("doc_id") or "")
    return _summary_view(candidate.node_id, metadata)


def _summary_view(node_id: str, metadata: dict[str, object]) -> dict[str, str]:
    return {
        "node_id": node_id,
        "file_name": str(metadata.get("file_name") or ""),
        "archive_path": str(metadata.get("archive_path") or ""),
        "heading_path": str(metadata.get("heading_path") or ""),
        "summary": str(metadata.get("summary") or ""),
        "snippet": str(metadata.get("text") or "")[:SNIPPET_CHARS],
    }


def _normalize_radius(radius: int | None) -> int:
    if radius is None:
        return 1
    return min(max(int(radius), 1), 2)


def _neighbor_views(metadata_index: object, node_id: str, metadata: dict[str, object], radius: int) -> list[dict[str, str]]:
    doc_id = str(metadata.get("doc_id") or "")
    chunk_index = metadata.get("chunk_index")
    if not doc_id or chunk_index is None:
        return []
    center = int(chunk_index)
    neighbors: list[dict[str, str]] = []
    lookup = metadata_index.doc_chunk_to_node_id
    for offset in range(1, radius + 1):
        for delta, relation in ((-offset, "previous"), (offset, "next")):
            neighbor_id = lookup.get((doc_id, center + delta))
            if neighbor_id is None or neighbor_id == node_id:
                continue
            item = _summary_view(neighbor_id, metadata_index.metadata_by_id.get(neighbor_id, {}))
            item["relation"] = relation
            neighbors.append(item)
    return neighbors


def _grep_markdown(
    docs_dir: Path,
    metadata_index: object,
    pattern: str,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int, int | None]:
    grouped: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for node_id, metadata in metadata_index.metadata_by_id.items():
        relative = _stored_markdown_relative(metadata)
        if not relative:
            continue
        grouped.setdefault(relative, []).append((str(node_id), metadata))

    by_document: dict[str, list[dict[str, object]]] = {}
    for relative in sorted(grouped):
        file_path = (docs_dir / relative).resolve()
        if not _is_under(file_path, docs_dir) or not file_path.is_file():
            continue
        matched: list[tuple[str, dict[str, object], int, str]] = []
        for node_id, metadata in grouped[relative]:
            text = str(metadata.get("text") or "")
            position = text.find(pattern)
            if position < 0:
                continue
            matched.append((node_id, metadata, position, text))
        matched.sort(key=lambda item: (_chunk_index_sort_key(item[1]), item[0]))
        document_matches: list[dict[str, object]] = []
        for node_id, metadata, position, text in matched:
            item = _summary_view(node_id, metadata)
            item["file_name"] = file_path.name
            item["file_path"] = str(file_path)
            item["chunk_index"] = _chunk_index_value(metadata)
            item["snippet"] = _match_snippet(text, position, len(pattern))
            document_matches.append(item)
        if document_matches:
            by_document[relative] = document_matches
    ordered = _interleave_documents(by_document)
    page = ordered[offset:offset + limit]
    next_offset = offset + len(page)
    omitted_count = max(len(ordered) - next_offset, 0)
    return page, omitted_count, next_offset if omitted_count else None


def _interleave_documents(by_document: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    """每个文档先返回一条，再返回各文档的下一条，避免一篇文档占满结果。"""
    if not by_document:
        return []
    ordered: list[dict[str, object]] = []
    rounds = max(len(items) for items in by_document.values())
    for round_index in range(rounds):
        for relative in sorted(by_document):
            items = by_document[relative]
            if round_index < len(items):
                ordered.append(items[round_index])
    return ordered


def _chunk_index_sort_key(metadata: dict[str, object]) -> tuple[int, int]:
    value = _chunk_index_value(metadata)
    if value is None:
        return (1, 0)
    return (0, value)


def _chunk_index_value(metadata: dict[str, object]) -> int | None:
    raw = metadata.get("chunk_index")
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _match_snippet(text: str, start: int, length: int) -> str:
    padding = max(SNIPPET_CHARS - length, 0) // 2
    left = max(start - padding, 0)
    right = min(start + length + padding, len(text))
    return text[left:right].replace("\n", " ")[:SNIPPET_CHARS]


def _dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "read_chunk": read_chunk,
    "get_dependencies": get_dependencies,
    "read_neighbors": read_neighbors,
    "grep_docs": grep_docs,
}

DEFAULT_QUESTION = ("我现在要接入ios订阅的最新上报接口，同时更具对应接口协议，提供java接入代码示例")
MAX_TOOL_ROUNDS = 20
# 只保留角色、目标、手段、顺序和硬边界。工具参数与返回字段以 TOOLS 为准。
SYSTEM_PROMPT = """\
你是企业知识库问答助手。

目标：依据知识库资料回答问题。

可用手段：search_knowledge_base 做语义检索，grep_docs 精确查找原词，read_neighbors 查看同文档前后文，get_dependencies 查看必要前置，read_chunk 读取片段全文。用法以各工具定义为准。

回应顺序：先检索，再依据已返回的结果决定是否继续检索、检索多少，或直接回答。证据够了就回答，并列出引用的文件名和 node_id。

硬边界：
- 只使用工具返回的资料和其中出现的 node_id。
- 检索量和是否继续由你判断，当前结果已经够回答就停止。
- 资料不足时说明无法从现有文档回答。
"""
HISTORY_BOUNDARY = """

最近几轮对话只用于理解指代和省略。证据必须来自本轮工具返回的资料，不能用历史回答代替检索结果。
"""
FOLLOW_UP_REWRITE_PROMPT = """\
你负责把追问改写成可以独立检索的一句问题。
你会看到按时间排列的用户提问，最后给出的是当前追问。
用这些提问补全省略的产品、接口、配置或对象。
不要回答，不要解释，不要加引号。
当前句已经完整时，原样输出。
只输出一句问题。
"""


def main() -> None:
    """用百炼兼容接口跑一轮工具调用，便于查看当前检索效果。"""
    import argparse

    from openai import OpenAI

    from rag.shared.logging_utils import configure_console_logging

    configure_console_logging()
    parser = argparse.ArgumentParser(description="用工具调用测试知识库 Agent")
    parser.add_argument("question", nargs="?", default=None, help="要测试的问题")
    parser.add_argument("--chat", action="store_true", help="进入多轮会话，追问会沿用上一轮主题")
    args = parser.parse_args()

    config = _get_query_config()
    if not config.llm.api_key:
        raise SystemExit("缺少 DASHSCOPE_API_KEY")
    client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
    model = os.getenv("MODEL") or config.answer.model
    session = Session(rewriter=lambda questions, current: rewrite_follow_up(client, model, questions, current))
    if args.chat:
        if args.question:
            _run_tool_loop(client, model, args.question, session)
        _chat_loop(client, model, session)
        return
    _run_tool_loop(client, model, args.question or DEFAULT_QUESTION, session)


def _chat_loop(client: object, model: str, session: Session) -> None:
    print("输入问题，空行或 exit 结束。")
    while True:
        try:
            question = input("问题> ").strip()
        except EOFError:
            return
        if question.casefold() in {"", "exit", "quit", "退出"}:
            return
        _run_tool_loop(client, model, question, session)


def _run_tool_loop(client: object, model: str, question: str, session: Session | None = None) -> str:
    active_session = session or Session()
    prepared = active_session.prepare(question)
    logger = logging.getLogger("rag.agent.session")
    logger.info(
        "问题路由 | route=%s | question=%s | resolved=%s",
        prepared.route,
        prepared.question,
        prepared.resolved_question,
    )
    if prepared.resolved_question != prepared.question:
        print(f"[补全] {prepared.resolved_question}")
    history = active_session.answer_messages()
    if prepared.route == CHAT:
        answer = _answer_without_tools(client, model, prepared.question)
    else:
        answer = _answer_with_tools(client, model, prepared.resolved_question, history)
    print(answer)
    active_session.record(prepared, answer)
    return answer


def _answer_without_tools(client: object, model: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是企业知识库问答助手。用户现在不是在查文档，请简短回答，不要编造文档内容。",
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return (response.choices[0].message.content or "").strip()


def rewrite_follow_up(client: object, model: str, questions: list[str], current: str) -> str:
    """用历史提问把追问改写成独立问题。失败时返回空字符串，交给规则补全。"""
    if not questions:
        return current
    numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    logger = logging.getLogger("rag.agent.session")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FOLLOW_UP_REWRITE_PROMPT},
                {"role": "user", "content": f"历史提问：\n{numbered}\n\n当前追问：{current}"},
            ],
            temperature=0,
            extra_body={"enable_thinking": False},
        )
    except Exception:
        logger.exception("追问改写失败")
        return ""
    text = (response.choices[0].message.content or "").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip().strip("\"“”")


def assemble_tool_messages(history: list[dict[str, str]], question: str) -> list[dict[str, object]]:
    """组装本轮消息：历史问答在前，当前问题在后，工具记录只追加在当前问题之后。"""
    system = SYSTEM_PROMPT + (HISTORY_BOUNDARY if history else "")
    messages: list[dict[str, object]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def _answer_with_tools(
    client: object,
    model: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    messages = assemble_tool_messages(history or [], question)
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return (message.content or "").strip()

        messages.append(message.model_dump(exclude_none=True))
        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                result = _dump({"error": "参数不是合法 JSON"})
            else:
                print(f"[工具] {name} {args}")
                try:
                    result = FUNCTIONS[name](**args)
                except KeyError:
                    result = _dump({"error": f"未知工具 {name}"})
                except Exception as exc:
                    result = _dump({"error": str(exc)})
            print(f"[结果] {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "工具调用轮次过多，已停止。"


if __name__ == "__main__":
    main()
