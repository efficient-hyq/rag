from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rag.config.common import default_storage_dir
from rag.preprocess.enriched_corpus import PrepareChunksConfig, prepare_enriched_corpus
from rag.shared.logging_utils import configure_console_logging, log_phase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析切分导出参数。默认路径都是项目根目录 storage 下的绝对路径。"""
    storage = default_storage_dir()
    parser = argparse.ArgumentParser(description="预处理文档并导出带图片说明的 chunk，不调用标注或向量接口")
    parser.add_argument("--input-dir",default=str(storage / "wiki-export"),help="原始 doc/docx 文档目录，默认是项目 storage/wiki-export")
    parser.add_argument("--storage-dir",default=str(storage),help="预处理、图片说明和 chunk 导出的根目录，默认是项目根目录下的 storage；相对路径按项目根目录解析")
    parser.add_argument("--captions", default=None, help="图片中文说明 JSON，默认是 <storage-dir>/image_captions.json")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理，直接使用已有 cleaned_markdown")
    return parser.parse_args(argv)


def main() -> None:
    """预处理文档、写入图片说明，并按当前切分规则导出 chunk。"""
    configure_console_logging()
    logger = logging.getLogger("rag.cli.prepare_chunks")
    args = parse_args()

    config = PrepareChunksConfig(
        input_dir=Path(args.input_dir),
        storage_dir=Path(args.storage_dir),
        captions_path=Path(args.captions) if args.captions else None,
        skip_preprocess=args.skip_preprocess,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
    )
    with log_phase(logger, "准备带图片说明的 chunk", input_dir=args.input_dir, storage_dir=str(config.storage_dir)):
        result = prepare_enriched_corpus(config)
    print(
        "切分准备完成: "
        f"markdown={result.markdown_count}, chunks={result.chunk_count}, "
        f"captioned_images={result.captioned_image_count}, missing_captions={result.missing_caption_count}, "
        f"chunks_path={result.chunks_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
