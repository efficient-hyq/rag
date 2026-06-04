# RAG 项目

一个面向本地知识库场景的 Python RAG 项目，覆盖文档预处理、离线建库、在线检索、双阶段重排和最终答案生成，适合技术文档、接口文档、Confluence/Word 导出文档的检索增强场景。

## 1. 项目概览

当前项目围绕两条主线构建：

- 离线链路：文档预处理 -> Markdown 加载 -> 语义切分 -> LLM 标注 -> Embedding 向量化 -> Chroma/BM25/Metadata 多路入库
- 在线链路：用户问题 -> Query Rewrite -> 多路召回 -> 规则粗排 -> 邻居扩展 -> LLM 精排 -> 答案生成 -> 来源输出

在线查询阶段统一使用 `qwen3.6-plus` 族模型完成改写、精排和最终答案生成。

## 2. 当前已实现能力

### 2.1 文档预处理

- 支持 `doc`、`docx` 输入。
- 支持 Confluence MHTML 风格导出文件解析。
- 支持 Word/Confluence 文档转 HTML、清洗 HTML、转换 Markdown。
- 支持图片资源抽取、路径重写、来源映射清单生成。

核心文件：

- [rag/preprocess/document_preprocessor.py](D:/ai/project/rag/rag/preprocess/document_preprocessor.py)
- [rag/cli/preprocess_docs.py](D:/ai/project/rag/rag/cli/preprocess_docs.py)

### 2.2 离线建库

- 读取清洗后的 Markdown 文档。
- 按 Markdown 语义块进行切分，尽量避免拆断代码块和表格。
- 使用 LLM 为 chunk 生成摘要、关键词、标签、类型、完整度等语义元数据。
- 生成正文向量和摘要向量两路 embedding。
- 写入 Chroma 向量库、BM25 索引和 `metadata.json`。
- 生成 chunk 预览页与检查点，支持断点续跑。
- `chunks_preview.html` 额外展示每个 chunk 的 `node_id`。

核心文件：

- [rag/indexing/index_builder.py](D:/ai/project/rag/rag/indexing/index_builder.py)
- [rag/indexing/document_loader.py](D:/ai/project/rag/rag/indexing/document_loader.py)
- [rag/indexing/markdown_chunker.py](D:/ai/project/rag/rag/indexing/markdown_chunker.py)
- [rag/indexing/semantic_annotator.py](D:/ai/project/rag/rag/indexing/semantic_annotator.py)
- [rag/indexing/embedding_client.py](D:/ai/project/rag/rag/indexing/embedding_client.py)
- [rag/indexing/storage_indexer.py](D:/ai/project/rag/rag/indexing/storage_indexer.py)
- [rag/indexing/preview_renderer.py](D:/ai/project/rag/rag/indexing/preview_renderer.py)
- [rag/cli/build_index.py](D:/ai/project/rag/rag/cli/build_index.py)

### 2.3 在线查询

- 支持 Query Rewrite 改写。
- 支持三路召回：
  - 正文向量召回
  - 摘要向量召回
  - BM25 关键词召回
- 支持候选聚合与去重。
- 支持邻居 chunk 扩展。
- 支持规则粗排与 LLM 精排双阶段重排。
- 支持基于 Top chunk 的最终答案生成。
- 支持输出答案、来源引用和 TopN 候选详情。

核心文件：

- [rag/retrieval/retriever.py](D:/ai/project/rag/rag/retrieval/retriever.py)
- [rag/retrieval/ranking.py](D:/ai/project/rag/rag/retrieval/ranking.py)
- [rag/retrieval/answer_generator.py](D:/ai/project/rag/rag/retrieval/answer_generator.py)
- [rag/retrieval/query_service.py](D:/ai/project/rag/rag/retrieval/query_service.py)
- [rag/retrieval/tokenization.py](D:/ai/project/rag/rag/retrieval/tokenization.py)
- [rag/cli/answer_query.py](D:/ai/project/rag/rag/cli/answer_query.py)

## 3. 项目结构

```text
rag/
├── AGENTS.md
├── README.md
├── rag/
│   ├── cli/
│   │   ├── answer_query.py
│   │   ├── build_index.py
│   │   └── preprocess_docs.py
│   ├── indexing/
│   │   ├── document_loader.py
│   │   ├── embedding_client.py
│   │   ├── index_builder.py
│   │   ├── markdown_chunker.py
│   │   ├── preview_renderer.py
│   │   ├── semantic_annotator.py
│   │   └── storage_indexer.py
│   ├── preprocess/
│   │   └── document_preprocessor.py
│   ├── retrieval/
│   │   ├── answer_generator.py
│   │   ├── query_service.py
│   │   ├── ranking.py
│   │   ├── retriever.py
│   │   └── tokenization.py
│   ├── shared/
│   │   ├── checkpoints.py
│   │   └── logging_utils.py
│   └── config/
│       ├── common.py
│       ├── indexing.py
│       └── retrieval.py
├── prompts/
├── docs/
├── tests/
└── storage/
```

这样调整后的原则是：

- `rag/cli/` 只放命令入口，不再使用含义模糊的 `main.py`
- `rag/preprocess/` 独立承载原始文档清洗链路
- `rag/indexing/` 专注离线建库
- `rag/retrieval/` 专注在线查询与答案生成
- `rag/shared/` 统一放日志与检查点等通用能力

## 4. 运行环境

### 4.1 基础要求

- Python 3.11 及以上
- Windows 环境
- 建议使用项目自带虚拟环境 `.venv`
- 文档预处理依赖 LibreOffice `soffice`

### 4.2 安装依赖

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 5. 配置说明

项目采用按流程拆分的环境变量配置：

- 离线建库配置定义在 [rag/config/indexing.py](D:/ai/project/rag/rag/config/indexing.py)
- 在线查询配置定义在 [rag/config/retrieval.py](D:/ai/project/rag/rag/config/retrieval.py)
- 共享路径、LLM、Embedding 配置定义在 [rag/config/common.py](D:/ai/project/rag/rag/config/common.py)

常用配置项：

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_DOCS_DIR` | 清洗后 Markdown 目录 | `./storage/cleaned_markdown` |
| `RAG_STORAGE_DIR` | 索引与检查点目录 | `./storage` |
| `RAG_ANNOTATOR_MODEL` | 标注模型 | `qwen3.6-plus` |
| `RAG_QUERY_REWRITE_MODEL` | 查询改写模型 | `qwen3.6-plus` |
| `RAG_RERANK_LLM_MODEL` | 精排模型 | `qwen3.6-plus` |
| `RAG_ANSWER_MODEL` | 答案生成模型 | `qwen3.6-plus` |
| `RAG_EMBEDDING_MODEL` | 向量模型 | `text-embedding-v4` |
| `RAG_ANSWER_ENABLED` | 是否启用答案生成 | `True` |

更完整的配置说明见：

- [docs/configuration.md](D:/ai/project/rag/docs/configuration.md)

## 6. 快速开始

### 6.1 预处理原始文档

```bash
.venv/Scripts/python.exe -m rag.cli.preprocess_docs --input-dir "C:/Users/heyuqin/Desktop/RAG_DATA"
```

### 6.2 构建离线索引

```bash
.venv/Scripts/python.exe -m rag.cli.build_index --docs-dir "./storage/cleaned_markdown" --storage-dir "./storage"
```

### 6.3 执行完整查询

```bash
.venv/Scripts/python.exe -m rag.cli.answer_query "我要接入谷歌订阅，现在需要如何处理"
```

输出包括：

- 改写结果
- 最终答案
- 引用来源（文件名、chunk 序号、node_id）
- TopN 候选 chunk 的路由来源、分数、文件名和 chunk 序号

## 7. 日志与进度

项目当前采用控制台双通道输出：

- 业务日志：通过 [rag/shared/logging_utils.py](D:/ai/project/rag/rag/shared/logging_utils.py) 输出到 `stderr`
- 进度展示：通过 [rag/shared/checkpoints.py](D:/ai/project/rag/rag/shared/checkpoints.py) 的 `print_progress()` 输出到 `stdout`

## 8. 关键产物说明

| 路径 | 说明 |
| --- | --- |
| `storage/checkpoints/chunks.jsonl` | 切分后的 chunk 快照 |
| `storage/checkpoints/annotations.jsonl` | 标注检查点 |
| `storage/checkpoints/embeddings_content.jsonl` | 正文向量检查点 |
| `storage/checkpoints/embeddings_summary.jsonl` | 摘要向量检查点 |
| `storage/checkpoints/manifest.json` | 本次建库参数快照 |
| `storage/bm25.pkl` | BM25 索引 |
| `storage/metadata.json` | 全量 chunk 元数据 |
| `storage/chunks_preview.html` | chunk 人工预览页面，包含 `node_id` |

## 9. 测试与验证

当前项目测试仍以轻量验证为主。

常用验证命令：

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m compileall rag tests
```

## 10. 相关文档

- [docs/技术分析文档.md](D:/ai/project/rag/docs/技术分析文档.md)
- [docs/项目流程文档.md](D:/ai/project/rag/docs/项目流程文档.md)
- [docs/configuration.md](D:/ai/project/rag/docs/configuration.md)
- [docs/RAG_设计方案.md](D:/ai/project/rag/docs/RAG_设计方案.md)
- [docs/RAG_优化优先级与排期建议.md](D:/ai/project/rag/docs/RAG_优化优先级与排期建议.md)

## 11. 文档维护要求

本次变更已涉及目录结构、运行入口、在线查询能力和预览产物，因此 README 与配套文档需要同步维护。后续若继续调整以下内容，也必须同步更新文档：

- 核心模块新增、删除或职责变化
- 离线建库或在线查询流程变化
- 配置项、运行命令或目录结构变化
- 新增答案生成、服务接口或评测链路
