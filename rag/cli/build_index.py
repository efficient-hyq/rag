from __future__ import annotations

import argparse
import logging

from rag.indexing.index_builder import build_offline_index
from rag.shared.logging_utils import configure_console_logging, log_phase


def main() -> None:
    """离线索引入口：默认按 Markdown 文档做增量删后重建。"""
    configure_console_logging()
    logger = logging.getLogger("rag.cli.build_index")
    parser = argparse.ArgumentParser(description="构建 RAG 离线索引（默认按 Markdown 文档增量重建）")
    parser.add_argument("--docs-dir", default="./storage/cleaned_markdown", help="清洗后的 Markdown 文档目录")
    parser.add_argument("--storage-dir", default="./storage", help="索引持久化目录")
    args = parser.parse_args()

    with log_phase(logger, "离线入库", docs_dir=args.docs_dir, storage_dir=args.storage_dir):
        result = build_offline_index(args.docs_dir, args.storage_dir)
    print(
        f"入库完成: rebuilt_chunks={result.node_count}, "
        f"metadata={result.metadata_path}, bm25={result.bm25_path}"
    )


if __name__ == "__main__":
    main()
