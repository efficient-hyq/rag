from __future__ import annotations

import argparse

from rag.retrieval.query_service import build_query_service, render_query_response
from rag.shared.logging_utils import configure_console_logging


def main() -> None:
    """命令行查询入口，执行完整的召回、重排和答案生成。"""
    configure_console_logging()
    parser = argparse.ArgumentParser(description="执行完整 RAG 查询并生成答案")
    parser.add_argument("question", nargs="?", default="我要接入iOS订阅，现在需要如何处理", help="用户问题")
    args = parser.parse_args()

    try:
        service = build_query_service()
        result = service.query(args.question)
    except Exception as exc:
        print(f"查询失败: {exc}")
        return

    print(render_query_response(result))


if __name__ == "__main__":
    main()
