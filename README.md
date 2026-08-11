# RAG 项目

一个面向本地知识库场景的 Python RAG 项目，覆盖文档预处理、离线建库、在线检索、双阶段重排和最终答案生成，适合技术文档、接口文档、Confluence/Word 导出文档的检索增强场景。

## 1. 项目概览

当前项目围绕两条主线构建：

- 离线链路：文档预处理 -> Markdown 切分与标题路径 -> LLM 标注 -> 文档依赖图 -> 多路向量化与入库 -> 工作区校验发布
- 在线链路：用户问题 -> Query Rewrite -> 多路召回 -> 动态 seed -> 一层必要依赖 -> 邻居扩展 -> LLM 精排 -> 答案与来源输出

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
- 默认按 `cleaned_markdown` Markdown 文档做增量建库。
- 未变化 Markdown 文档复用现有标注、向量检查点和索引元数据；只有标注或依赖图输出协议对应的 prompt 版本变化时才强制全量重建。
- 标注和依赖聚合的成功检查点按内容及 prompt 版本复用，不再把模型名作为命中条件；记录中的 `model` 字段保留实际生成模型，切换模型不会重复消耗 token。
- 变更 Markdown 文档会先删除旧 `node_id`，再重建该文档全部 chunk。
- 已删除 Markdown 文档会自动清理旧向量、检查点和 metadata 分片。
- 按 Markdown 语义块进行切分，尽量避免拆断代码块和表格。
- 由程序生成 `heading_path`，使用 LLM 为 chunk 生成摘要、关键词和自由标注的 `dependency_topics` 等语义元数据。
- `dependency_topics` 不使用固定注册表，只做清洗、去重和数量限制；它是可选解释信息，不参与文档增量失效或依赖边有效性判断。
- 按单个 Markdown 文档聚合全量 node 直接依赖图，并校验文档状态、node 覆盖和关系字段。
- 生成正文向量和摘要向量两路 embedding。
- 可选生成 `dependency_vec`，只用于定位 source node，不直接作为答案证据。
- 写入 Chroma 向量库、BM25 索引和 `metadata.json`。
- 在固定 `storage/index_workspace/` 中构建并保留检查点，全部校验成功后发布查询产物并原子更新根 `manifest.json`。
- 生成 chunk 预览页与检查点，支持断点续跑。
- `chunk_previews/` 按文档输出预览页，每个 chunk 展示 `node_id`。

核心文件：

- [rag/indexing/index_builder.py](D:/ai/project/rag/rag/indexing/index_builder.py)
- [rag/indexing/document_loader.py](D:/ai/project/rag/rag/indexing/document_loader.py)
- [rag/indexing/markdown_chunker.py](D:/ai/project/rag/rag/indexing/markdown_chunker.py)
- [rag/indexing/semantic_annotator.py](D:/ai/project/rag/rag/indexing/semantic_annotator.py)
- [docs/dependency_topics规范.md](D:/ai/project/rag/docs/dependency_topics规范.md)
- [rag/indexing/embedding_client.py](D:/ai/project/rag/rag/indexing/embedding_client.py)
- [rag/indexing/storage_indexer.py](D:/ai/project/rag/rag/indexing/storage_indexer.py)
- [rag/indexing/dependency_graph.py](D:/ai/project/rag/rag/indexing/dependency_graph.py)
- [rag/indexing/publication.py](D:/ai/project/rag/rag/indexing/publication.py)
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
- 支持规则优先、LLM 兜底的依赖召回路由。
- 支持按最高分 70% 动态选择 seed，并扩展一层必要依赖。
- 依赖候选按 `node_id` 从 metadata 精确取证，支持同文档定向缺失兜底。
- 支持规则粗排与 LLM 精排双阶段重排。
- 支持按“全部 seed 在前、必要依赖在后”组装上下文，答案生成阶段不再固定 TopK 截断。
- 支持输出答案、来源引用和 TopN 候选详情。

核心文件：

- [rag/retrieval/retriever.py](D:/ai/project/rag/rag/retrieval/retriever.py)
- [rag/retrieval/ranking.py](D:/ai/project/rag/rag/retrieval/ranking.py)
- [rag/retrieval/answer_generator.py](D:/ai/project/rag/rag/retrieval/answer_generator.py)
- [rag/retrieval/query_service.py](D:/ai/project/rag/rag/retrieval/query_service.py)
- [rag/retrieval/tokenization.py](D:/ai/project/rag/rag/retrieval/tokenization.py)
- [rag/retrieval/dependency_recall.py](D:/ai/project/rag/rag/retrieval/dependency_recall.py)
- [rag/cli/answer_query.py](D:/ai/project/rag/rag/cli/answer_query.py)

### 2.4 隐式业务依赖召回

- chunk 由 LLM 自由标注可选依赖主题，并由 Markdown 解析器生成标题路径。
- 按单个文档聚合全部 `node_id` 的直接依赖关系，没有依赖的 node 使用空数组。
- 普通召回命中 seed node 后，根据 `node_dependencies` 直接补充同文档依赖 node。
- `node_id`、关系类型、置信度和原文依据构成依赖召回主链路；topic 仅用于可选展示和向量文本补充，缺失时不影响建边和取证。
- 只在线使用 `status=success` 且 node 数量完整的文档卡片；默认过滤非必要边和置信度低于 `0.7` 的边。
- 单 seed 最多补 5 个依赖，最多 8 个 seed，上下文总安全上限为 12；被裁剪的必要依赖会写入日志。

完整设计见：[docs/隐式业务依赖漏召回解决方案.md](D:/ai/project/rag/docs/隐式业务依赖漏召回解决方案.md)

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
│   │   ├── dependency_graph.py
│   │   ├── dependency_topics.py
│   │   ├── embedding_client.py
│   │   ├── index_builder.py
│   │   ├── markdown_chunker.py
│   │   ├── preview_renderer.py
│   │   ├── publication.py
│   │   ├── semantic_annotator.py
│   │   └── storage_indexer.py
│   ├── preprocess/
│   │   └── document_preprocessor.py
│   ├── retrieval/
│   │   ├── answer_generator.py
│   │   ├── dependency_recall.py
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
| `RAG_DEPENDENCY_GRAPH_MODEL` | 文档依赖图聚合模型 | `qwen3.6-plus` |
| `RAG_DEPENDENCY_RECALL_ENABLED` | 是否启用在线依赖扩展 | `True` |
| `RAG_DEPENDENCY_VECTOR_ENABLED` | 是否生成并查询可选依赖向量 | `False` |
| `RAG_QUERY_REWRITE_MODEL` | 查询改写模型 | `qwen3.6-plus` |
| `RAG_RERANK_LLM_MODEL` | 精排模型 | `qwen3.6-plus` |
| `RAG_ANSWER_MODEL` | 答案生成模型 | `qwen3.6-plus` |
| `RAG_EMBEDDING_MODEL` | 向量模型 | `text-embedding-v4` |
| `RAG_ANSWER_ENABLED` | 是否启用答案生成 | `True` |

完整参数手册见下方文档，其中包含全部 CLI 参数、46 个唯一环境变量、准确默认值、覆盖优先级、必需配置和降级行为：

- [docs/configuration.md](D:/ai/project/rag/docs/configuration.md)

特别注意：当前 Embedding 配置不会回退到 `DASHSCOPE_API_KEY` 或 `DASHSCOPE_BASE_URL`，正常建库与查询需要单独设置 `RAG_EMBEDDING_API_KEY`；`RAG_EMBEDDING_BATCH_SIZE` 的实际默认值为 `20`。

## 6. 快速开始

### 6.1 预处理原始文档

```bash
.venv/Scripts/python.exe -m rag.cli.preprocess_docs --input-dir "C:/Users/heyuqin/Desktop/RAG_DATA"
```

### 6.2 构建离线索引

```bash
.venv/Scripts/python.exe -m rag.cli.build_index --docs-dir "./storage/cleaned_markdown" --storage-dir "./storage"
```

`build_index` 默认按 Markdown 文档增量执行，并始终使用固定的 `storage/index_workspace/`。新增和变更文档会整体重建，未变化文档直接复用该工作区中的标注、向量和依赖聚合检查点，删除文档会从工作区索引中排除；全部产物校验成功后才更新根 `manifest.json`，构建失败不会改变在线最终目录。旧式根目录存储首次迁移、依赖卡片不完整，或标注/依赖图输出协议对应的 prompt 版本变化时，会全量重建当前文档。自由 topic 的新增、改名或写法变化不会单独触发全量重建。

已有工作区若仍使用旧版带模型 key，可先执行以下命令原地迁移检查点；该操作只修改 `storage/index_workspace/checkpoints/`，不会调用模型：

```bash
.venv/Scripts/python.exe -m rag.cli.build_index --storage-dir "./storage" --migrate-workspace-checkpoints
```

根清单示例：

```json
{
  "schema_version": "index_manifest_v1",
  "final_index_dir": "published_index-<publish_token>",
  "embedding_model": "text-embedding-v4",
  "published_at": "2026-08-05T04:00:00Z"
}
```

在线查询只解析 `final_index_dir`，不会读取 `index_workspace`。发布目录名中的 token 仅用于避免 Windows 下 Chroma 文件锁阻塞新结果发布，不进入标注、向量、BM25、metadata 或依赖图数据。

根清单同时记录建库使用的 `embedding_model`。查询服务初始化时会将当前 `RAG_EMBEDDING_MODEL` 与该值严格比较；缺失或不一致会在问题编码前直接报错，不允许跨模型向量检索。切换 Embedding 模型必须全量重建并重新发布索引。

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
| `storage/manifest.json` | 记录查询使用的 `final_index_dir` |
| `storage/index_workspace/checkpoints/chunks.jsonl` | 切分后的 chunk 快照 |
| `storage/index_workspace/checkpoints/annotations.jsonl` | 标注检查点 |
| `storage/index_workspace/checkpoints/embeddings_*.jsonl` | 正文和摘要向量检查点 |
| `storage/index_workspace/checkpoints/dependency_cards.jsonl` | 文档依赖聚合检查点 |
| `storage/index_workspace/checkpoints/document_index_state.json` | Markdown 文档级索引状态清单 |
| `storage/index_workspace/metadata_docs/` | 按 Markdown 文档分片的 metadata |
| `storage/index_workspace/chunk_previews/` | 按文档输出的 chunk 人工预览页面 |
| `storage/<final_index_dir>/bm25.pkl` | 在线查询使用的 BM25 索引 |
| `storage/<final_index_dir>/metadata.json` | 在线查询使用的全量 chunk 元数据 |
| `storage/<final_index_dir>/chroma/` | 在线查询使用的正文、摘要和可选依赖向量集合 |
| `storage/<final_index_dir>/dependency_cards.json` | 在线查询使用的文档级全量 node 直接依赖图 |

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
- [docs/隐式业务依赖漏召回解决方案.md](D:/ai/project/rag/docs/隐式业务依赖漏召回解决方案.md)

## 11. 文档维护要求

本次变更已涉及目录结构、运行入口、在线查询能力和预览产物，因此 README 与配套文档需要同步维护。后续若继续调整以下内容，也必须同步更新文档：

- 核心模块新增、删除或职责变化
- 离线建库或在线查询流程变化
- 配置项、运行命令或目录结构变化
- 新增答案生成、服务接口或评测链路
