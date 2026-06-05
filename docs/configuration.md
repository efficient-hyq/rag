# 配置说明

本文档说明当前 RAG 项目的环境变量配置、作用范围与主要输出产物。若 `rag/config/` 下的配置模块发生变更，必须同步更新本文件。

## 1. 配置加载入口

当前项目按流程拆分加载运行配置：

- [rag/config/indexing.py](D:/ai/project/rag/rag/config/indexing.py) 中的 `BuildIndexConfig.from_env()`
- [rag/config/retrieval.py](D:/ai/project/rag/rag/config/retrieval.py) 中的 `QueryConfig.from_env()`
- [rag/config/common.py](D:/ai/project/rag/rag/config/common.py) 中的共享配置加载函数

覆盖范围包括：

- 离线建库
- 在线查询
- Query Rewrite
- LLM 精排
- 最终答案生成

## 2. 基础配置

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_DOCS_DIR` | 清洗后的 Markdown 文档目录 | `./storage/cleaned_markdown` |
| `RAG_STORAGE_DIR` | 索引、检查点和预览输出目录 | `./storage` |
| `RAG_CHUNK_SIZE` | chunk 大小 | `512` |
| `RAG_CHUNK_OVERLAP` | chunk 重叠大小 | `100` |

说明：

- `build_index` 默认按 `RAG_DOCS_DIR` 下的 Markdown 文档做文档级增量建库。
- 未变化文档复用现有标注、向量检查点和 metadata 分片。
- 新增或变更文档会重建该文档全部 chunk；删除文档会清理旧 `node_id`。

## 3. LLM 配置

当前标注、改写、精排和答案生成都通过 OpenAI 兼容接口调用。

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | LLM API Key | 无默认值，必须外部注入 |
| `DASHSCOPE_BASE_URL` | LLM 接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `RAG_ANNOTATOR_MODEL` | 离线标注模型 | `qwen3.6-plus` |
| `RAG_QUERY_REWRITE_MODEL` | 查询改写模型 | `qwen3.6-plus` |
| `RAG_RERANK_LLM_MODEL` | 精排模型 | `qwen3.6-plus` |
| `RAG_ANSWER_MODEL` | 最终答案生成模型 | `qwen3.6-plus` |
| `RAG_ANNOTATION_WORKERS` | 标注并发数 | `5` |
| `RAG_ANNOTATION_PROMPT_PATH` | 标注提示词文件路径 | `prompts/annotation_v2.md` |
| `RAG_ANNOTATION_PROMPT_VERSION` | 标注提示词版本号 | `annotation_v2` |

说明：

- 查询侧默认全部使用 `qwen3.6-plus`
- 若缺少 `DASHSCOPE_API_KEY`，改写、精排和答案生成会退化或关闭

## 4. 向量化配置

当前向量化由 [rag/indexing/embedding_client.py](D:/ai/project/rag/rag/indexing/embedding_client.py) 执行。

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_EMBEDDING_API_KEY` | 向量化 API Key，未设置时回退到 `DASHSCOPE_API_KEY` | 空 |
| `RAG_EMBEDDING_BASE_URL` | 向量接口地址，未设置时回退到 `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `RAG_EMBEDDING_MODEL` | 向量模型名称 | `text-embedding-v4` |
| `RAG_EMBEDDING_BATCH_SIZE` | 向量化批大小 | `10` |

## 5. 在线召回配置

以下配置影响 [rag/retrieval/retriever.py](D:/ai/project/rag/rag/retrieval/retriever.py) 的多路召回行为。

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_RETRIEVAL_CONTENT_TOP_K` | 原始 query 的正文向量召回数量 | `8` |
| `RAG_RETRIEVAL_SUMMARY_TOP_K` | 原始 query 的摘要向量召回数量 | `6` |
| `RAG_RETRIEVAL_BM25_TOP_K` | 原始 query 的 BM25 召回数量 | `8` |
| `RAG_RETRIEVAL_REWRITE_ENABLED` | 是否启用 Query Rewrite | `true` |
| `RAG_RETRIEVAL_REWRITE_LIMIT` | 改写问题数量上限 | `3` |
| `RAG_RETRIEVAL_REWRITE_CONTENT_TOP_K` | 改写 query 的正文向量召回数量 | `4` |
| `RAG_RETRIEVAL_REWRITE_SUMMARY_TOP_K` | 改写 query 的摘要向量召回数量 | `3` |
| `RAG_RETRIEVAL_REWRITE_BM25_TOP_K` | 改写 query 的 BM25 召回数量 | `4` |
| `RAG_RETRIEVAL_NEIGHBOR_ENABLED` | 是否启用邻居扩展 | `true` |
| `RAG_RETRIEVAL_NEIGHBOR_RADIUS` | 相邻扩展半径 | `1` |
| `RAG_RETRIEVAL_CENTER_TOP_K` | 做邻居扩展的中心 chunk 数 | `5` |

## 6. 重排与答案生成配置

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_RERANK_LLM_ENABLED` | 是否启用 LLM 精排 | `true` |
| `RAG_RERANK_LLM_TOP_N` | 进入 LLM 精排的候选数量 | `10` |
| `RAG_RERANK_FINAL_TOP_N` | 最终返回的候选数量 | `5` |
| `RAG_ANSWER_ENABLED` | 是否启用答案生成 | `true` |
| `RAG_ANSWER_CONTEXT_TOP_K` | 进入答案生成的上下文数量 | `4` |
| `RAG_ANSWER_MAX_CONTEXT_CHARS` | 单个上下文最大截断字符数 | `1200` |

## 7. 常用配置示例

### 7.1 最小可用本地配置

```bash
set DASHSCOPE_API_KEY=你的LLMKey
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set RAG_EMBEDDING_API_KEY=你的向量Key
set RAG_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set RAG_DOCS_DIR=./storage/cleaned_markdown
set RAG_STORAGE_DIR=./storage
```

### 7.2 调大召回与答案上下文

```bash
set RAG_RETRIEVAL_CONTENT_TOP_K=12
set RAG_RETRIEVAL_SUMMARY_TOP_K=10
set RAG_RETRIEVAL_BM25_TOP_K=12
set RAG_ANSWER_CONTEXT_TOP_K=5
set RAG_RERANK_FINAL_TOP_N=8
```

### 7.3 关闭改写与精排

```bash
set RAG_RETRIEVAL_REWRITE_ENABLED=false
set RAG_RERANK_LLM_ENABLED=false
```

### 7.4 关闭答案生成，仅看检索结果

```bash
set RAG_ANSWER_ENABLED=false
```

## 8. 输出文件说明

### 8.1 预处理阶段

| 文件或目录 | 说明 |
| --- | --- |
| `storage/converted_html/` | 中间 HTML 文件 |
| `storage/cleaned_markdown/` | 清洗后的 Markdown 文件 |
| `storage/cleaned_assets/` | 抽取的图片资源 |
| `storage/cleaned_markdown/source_manifest.json` | 原始文件与 Markdown 映射清单 |

### 8.2 离线建库阶段

| 文件或目录 | 说明 |
| --- | --- |
| `storage/checkpoints/chunks.jsonl` | 切分后的 chunk 快照 |
| `storage/checkpoints/annotations.jsonl` | 标注检查点 |
| `storage/checkpoints/embeddings_content.jsonl` | 正文向量检查点 |
| `storage/checkpoints/embeddings_summary.jsonl` | 摘要向量检查点 |
| `storage/checkpoints/manifest.json` | 本次建库参数快照 |
| `storage/checkpoints/document_index_state.json` | Markdown 文档级索引状态清单 |
| `storage/chroma/` | Chroma 持久化目录 |
| `storage/bm25.pkl` | BM25 索引文件 |
| `storage/metadata_docs/` | 按 Markdown 文档分片的 metadata |
| `storage/metadata.json` | 兼容查询侧的全量 chunk 元数据快照 |
| `storage/chunk_previews/` | 按文档输出的 chunk 预览页面，包含 `node_id` |

## 9. 风险提示

当前配置层需要重点注意：

- LLM 与 Embedding 的 Base URL 不要混用
- 查询侧虽然默认使用 `qwen3.6-plus`，但仍依赖兼容接口可用性
- 若关闭 `RAG_ANSWER_ENABLED`，命令行会退回到基于检索结果的摘要输出
- 若 `metadata_docs/`、`metadata.json` 与 Chroma 内容不一致，优先检查 `document_index_state.json` 中的 `node_ids`
