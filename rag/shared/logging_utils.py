from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


def configure_console_logging(level: int = logging.INFO) -> None:
    """把日志输出到控制台的 stderr，避免影响 stdout 进度输出。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


@contextmanager
def log_phase(logger: logging.Logger, phase: str, **fields: object) -> Iterator[None]:
    """记录一个阶段的开始、结束、耗时和异常。"""
    start = perf_counter()
    logger.info("%s开始%s", phase, _format_fields(fields))
    try:
        yield
    except Exception:
        logger.exception("%s失败%s", phase, _format_fields(fields))
        raise
    else:
        elapsed = perf_counter() - start
        logger.info("%s完成%s | 耗时=%.2fs", phase, _format_fields(fields), elapsed)


def _format_fields(fields: dict[str, object]) -> str:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    if not parts:
        return ""
    return " | " + "; ".join(parts)
