# RAG 项目

一个面向本地知识库场景的 Python RAG 项目，覆盖文档预处理、离线建库、在线检索、双阶段重排和最终答案生成，适合技术文档、接口文档、Confluence/Word 导出文档的检索增强场景。

## 1. 项目概览

当前项目围绕两条主线构建：

- 离线链路：文档预处理 -> 独立切分导出 chunk -> 按 chunk 标注、上下文嵌入和向量化 -> 文档依赖图 -> 工作区校验发布
- 在线链路：用户问题 -> Query Rewrite -> 多路召回 -> 动态 seed -> 一层必要依赖 -> 邻居扩展 -> LLM 精排 -> 答案与来源输出

在线查询阶段统一使用 `qwen3.6-plus` 族模型完成改写、精排和最终答案生成。

## 2. 当前已实现能力

### 2.1 文档预处理

- 支持 `doc`、`docx` 输入。
- 支持 Confluence MHTML 风格导出文件解析。
- 支持 Word/Confluence 文档转 HTML、清洗 HTML、转换 Markdown。
- 支持图片资源抽取、路径重写、来源映射清单生成。
- 支持把已有中文图片说明写进图片同一行，再按当前切分规则导出 chunk。chunk 只保留预处理后的文档相对路径和归档目录，不保留原始 doc/docx 路径。

核心文件：

- [rag/preprocess/document_preprocessor.py](D:/ai/project/rag/rag/preprocess/document_preprocessor.py)
- [rag/preprocess/image_captions.py](D:/ai/project/rag/rag/preprocess/image_captions.py)
- [rag/preprocess/enriched_corpus.py](D:/ai/project/rag/rag/preprocess/enriched_corpus.py)
- [rag/cli/preprocess_docs.py](D:/ai/project/rag/rag/cli/preprocess_docs.py)
- [rag/cli/prepare_chunks.py](D:/ai/project/rag/rag/cli/prepare_chunks.py)

### 2.2 离线建库

- 读取 `prepare_chunks` 导出的 `prepared_chunks/chunks.jsonl`，建库阶段不再切分 Markdown。
- 按 chunk 指纹做增量：新增和内容变化的 chunk 重新标注并嵌入，消失的 chunk 从索引删除。
- 未变化 chunk 复用标注和向量检查点；标注或依赖图输出协议对应的 prompt 版本变化时才全量重跑。
- 标注和依赖聚合的成功检查点按内容及 prompt 版本复用，不再把模型名作为命中条件；记录中的 `model` 字段保留实际生成模型，切换模型不会重复消耗 token。
- 正文向量和 BM25 使用带归档路径的上下文，用来区分同一上级目录下的不同业务线，例如 Google 订阅和支付网关。
- 由切分阶段生成 `heading_path` 和 `archive_path`。`heading_path` 按标题级别入栈，文档 H1 会进入后续章节；覆盖整节的 chunk 使用该节标题。超长原子章节超过上限后按常规规则落盘，只有标题的碎片会并入相邻 chunk。
- 使用 LLM 为 chunk 生成摘要、关键词、`dependency_topics`，以及当前段身份 `subject` 和替别人定下的规矩 `stated_rules`。
- `dependency_topics` 不使用固定注册表，只做清洗、去重和数量限制；它是可选解释信息，不参与文档增量失效或依赖边有效性判断。
- 按单个 Markdown 文档把全部 `subject` 和 `stated_rules` 一次交给模型整理直接依赖。没有规矩的文档不调用模型。没有被约束到的段落由程序补成空数组。
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
  - BM25 关键词召回。分词保留词频，并过滤停用词和标点；得分为 0 或负数的命中不会进入排序
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
| `RAG_DOCS_DIR` | 预处理 Markdown 目录，供读取原文 | 项目 `storage/enriched_markdown`，不存在时用 `cleaned_markdown` |
| `RAG_STORAGE_DIR` | 索引、预处理和 chunk 根目录 | 项目根目录下的 `storage` |
| `RAG_ANNOTATOR_MODEL` | 标注模型 | `qwen3.6-plus` |
| `RAG_CHUNK_CONTEXT_MODEL` | 片段定位上下文模型 | 无，必须设置 |
| `RAG_DEPENDENCY_GRAPH_MODEL` | 文档依赖图聚合模型 | `qwen3.6-plus` |
| `RAG_DEPENDENCY_RECALL_ENABLED` | 是否启用在线依赖扩展 | `True` |
| `RAG_DEPENDENCY_VECTOR_ENABLED` | 是否生成并查询可选依赖向量 | `False` |
| `RAG_QUERY_REWRITE_MODEL` | 查询改写模型 | `qwen3.6-plus` |
| `RAG_RERANK_LLM_MODEL` | 精排模型 | `qwen3.6-plus` |
| `RAG_ANSWER_MODEL` | 答案生成模型 | `qwen3.6-plus` |
| `RAG_EMBEDDING_MODEL` | 向量模型 | `text-embedding-v4` |
| `RAG_ANSWER_ENABLED` | 是否启用答案生成 | `True` |

完整参数手册见下方文档，其中包含全部 CLI 参数、50 个唯一环境变量、准确默认值、覆盖优先级、必需配置和降级行为：

- [docs/configuration.md](D:/ai/project/rag/docs/configuration.md)

特别注意：当前 Embedding 配置不会回退到 `DASHSCOPE_API_KEY` 或 `DASHSCOPE_BASE_URL`，正常建库与查询需要单独设置 `RAG_EMBEDDING_API_KEY`；`RAG_EMBEDDING_BATCH_SIZE` 的实际默认值为 `20`。

## 6. 快速开始

### 6.1 预处理原始文档

```bash
.venv/Scripts/python.exe -m rag.cli.preprocess_docs --input-dir "D:/ai/project/rag/storage/wiki-export"
```

### 6.2 写入图片说明并导出 chunk

这一步不调用标注、摘要、关键词或向量接口。它复用预处理结果，把 `storage/image_captions.json` 中的中文说明接到对应图片同一行，写出 `storage/enriched_markdown/`，再按当前 `chunk_size=512`、`chunk_overlap=100` 导出 `storage/prepared_chunks/chunks.jsonl`。

图片和说明放在同一行，是为了让现有按行切分不会把二者拆开。`cleaned_markdown` 保持预处理原样，重复预处理不会冲掉说明。chunk 记录包含 `node_id`、正文、`heading_path`、`chunk_index`、`token_size`、文件名、预处理文档相对路径、归档目录和图片说明，不包含原始文档路径。

不传路径时，输入目录是项目 `storage/wiki-export`，存储根目录是项目根下的 `storage`。这两个默认值都是绝对路径，不随当前工作目录变化。显式传入的相对 `--storage-dir` 仍按项目根目录解析。

```bash
.venv/Scripts/python.exe -m rag.cli.prepare_chunks
```

已经完成预处理、只想重新套用图片说明时，加上 `--skip-preprocess`。

### 6.3 构建离线索引

```bash
.venv/Scripts/python.exe -m rag.cli.build_index
```

`build_index` 读取 `storage/prepared_chunks/chunks.jsonl`，不再扫描 Markdown，也不再切分。它先给每个 chunk 写定位上下文，再让标注、正文向量和 BM25 使用同一份检索文本，然后生成摘要向量和依赖图。摘要、关键词和标签吸收定位说明，依赖主题只依据正文。chunk 原文保持不变。运行前需要设置 `RAG_CHUNK_CONTEXT_MODEL`。存储目录同样固定在项目根目录的 `storage/index_workspace/`。未变化 chunk 复用检查点；消失的 chunk 会从工作区索引中删除。全部产物校验成功后才更新根 `manifest.json`，构建失败不会改变在线最终目录。依赖卡片不完整，或标注/依赖图输出协议对应的 prompt 版本变化时，会重跑受影响的 chunk。自由 topic 的新增、改名或写法变化不会单独触发全量重建。

切换到这套 chunk 输入后，需要重新执行 `prepare_chunks` 和 `build_index`，旧的文档级索引不会自动带上归档路径。

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

### 6.4 执行完整查询

```bash
.venv/Scripts/python.exe -m rag.cli.answer_query "我要接入谷歌订阅，现在需要如何处理"
```

输出包括：

- 改写结果
- 最终答案
- 引用来源（文件名、chunk 序号、node_id）
- TopN 候选 chunk 的路由来源、分数、文件名和 chunk 序号

### 6.5 测试 Agent 工具调用

```bash
.venv/Scripts/python.exe -m rag.agent.tools "我要接入谷歌订阅，现在需要如何处理"
.venv/Scripts/python.exe -m rag.agent.tools --chat
```

这个入口调用 `search_knowledge_base`、`grep_docs`、`read_neighbors`、`get_dependencies` 和 `read_chunk`。前四个工具只返回摘要和短片段，全文通过 `read_chunk` 按需读取。`--chat` 进入多轮会话：闲聊不检索。追问会先把最近最多 8 轮用户提问交给模型，改写成可以独立检索的问题；改写失败时仍用产品名和接口名做规则补全。回答阶段只带最近 3 轮问答。会话里保存完整回答，放进上下文时才把每条截断到 400 字，并且不包含上一轮的工具调用记录。本轮工具请求和工具结果只追加在当前问题后面。历史回答不能代替本轮检索证据。系统提示词只规定角色、目标和检索边界，工具参数以工具定义为准；检索量和是否继续由模型判断，证据够了就停止。需要已设置 `DASHSCOPE_API_KEY`。模型默认使用 `RAG_ANSWER_MODEL`，也可以用环境变量 `MODEL` 覆盖。

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
