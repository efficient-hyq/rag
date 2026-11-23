from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag.indexing.index_builder import build_offline_index
from rag.shared.logging_utils import configure_console_logging, log_phase


def main() -> None:
    """离线索引入口：读取已切分 chunk，按 chunk 做增量标注和入库。"""
    configure_console_logging()
    logger = logging.getLogger("rag.cli.build_index")
    parser = argparse.ArgumentParser(description="构建 RAG 离线索引（读取 prepare_chunks 导出的 chunk，不再切分）")
    parser.add_argument("--chunks", default=None, help="已切分的 chunks.jsonl，默认是项目 storage/prepared_chunks/chunks.jsonl")
    parser.add_argument("--storage-dir", default=None, help="索引根目录，默认是项目根目录下的 storage；相对路径按项目根目录解析")
    args = parser.parse_args()

    with log_phase(logger, "离线入库", chunks=args.chunks or "", storage_dir=args.storage_dir or ""):
        result = build_offline_index(args.chunks, args.storage_dir)
    print(
        f"入库完成: rebuilt_chunks={result.node_count}, "
        f"metadata={result.metadata_path}, bm25={result.bm25_path}"
    )


if __name__ == "__main__":
    main()
