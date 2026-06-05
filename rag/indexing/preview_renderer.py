from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any


def write_chunk_preview(nodes: list[Any], storage_dir: str | Path) -> Path:
    """导出中文友好的 chunk 预览页面，便于人工检查切分质量。"""
    output_path = Path(storage_dir) / "chunks_preview.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = _group_nodes_by_doc_id(nodes)
    output_path.write_text(_render_preview_html(groups), encoding="utf-8")
    return output_path


def write_document_chunk_previews(nodes: list[Any], storage_dir: str | Path) -> list[Path]:
    """按 Markdown 文档导出 chunk 预览页，便于增量重建后定向检查。"""
    output_root = Path(storage_dir) / "chunk_previews"
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for doc_id, doc_nodes in _group_nodes_by_doc_id(nodes).items():
        output_path = output_root / f"{_safe_doc_file_name(doc_id)}.html"
        output_path.write_text(
            _render_preview_html({doc_id: doc_nodes}),
            encoding="utf-8",
        )
        paths.append(output_path)
    return paths


def _group_nodes_by_doc_id(nodes: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}) or {})
        doc_id = str(
            metadata.get("cleaned_markdown_relative_path")
            or metadata.get("doc_id")
            or metadata.get("file_path")
            or "unknown"
        )
        groups[doc_id].append(node)
    return dict(groups)


def _render_preview_html(groups: dict[str, list[Any]]) -> str:
    sections = []
    for doc_id, doc_nodes in groups.items():
        cards = "\n".join(_render_node(node) for node in doc_nodes)
        sections.append(
            f"""
            <section class="document">
              <h2>{escape(doc_id)}</h2>
              <div class="chunks">{cards}</div>
            </section>
            """
        )

    html = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chunk 切分预览</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #667085;
      --line: #d7d9de;
      --accent: #1f7a5c;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      line-height: 1.65;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 247, 244, 0.96);
      padding: 16px 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px 24px 48px;
    }}
    .document {{
      margin: 0 0 28px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      overflow-wrap: anywhere;
    }}
    .chunks {{
      display: grid;
      gap: 12px;
    }}
    .chunk {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .badge {{
      color: var(--accent);
      font-weight: 700;
    }}
    .mono {{
      font-family: Consolas, "Courier New", monospace;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font: inherit;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Chunk 切分预览</h1>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    return html


def _safe_doc_file_name(doc_id: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", doc_id.replace("\\", "/").replace("/", "__")).strip("_")
    if not stem:
        stem = "unknown"
    suffix = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12]
    return f"{stem[:80]}-{suffix}"


def _render_node(node: Any) -> str:
    metadata = dict(getattr(node, "metadata", {}) or {})
    node_id = str(getattr(node, "node_id", None) or getattr(node, "id_", None) or "")
    index = metadata.get("chunk_index", "")
    token_size = metadata.get("token_size", "")
    file_name = metadata.get("file_name", "")
    coherence = metadata.get("coherence", "")
    summary = metadata.get("summary", "")
    keywords = ", ".join(str(item) for item in metadata.get("keywords", []) if item)
    text = str(getattr(node, "text", ""))
    return f"""
    <article class="chunk">
      <div class="meta">
        <span class="badge">chunk-{escape(str(index))}</span>
        <span class="mono">node_id：{escape(node_id)}</span>
        <span>token 估算：{escape(str(token_size))}</span>
        <span>文件：{escape(str(file_name))}</span>
        <span>完整度：{escape(str(coherence))}</span>
      </div>
      <div class="meta">
        <span>摘要：{escape(str(summary))}</span>
        <span>关键词：{escape(keywords)}</span>
      </div>
      <pre>{escape(text)}</pre>
    </article>
    """
