# RAG 参数与默认值说明

本文档以当前代码为准，完整说明 RAG 项目可传入的命令行参数、环境变量、默认值、作用范围、覆盖优先级和运行时行为。

参数来源对应以下实现：

- 命令行入口：`rag/cli/` 与 `rag/preprocess/document_preprocessor.py`
- 共享配置：`rag/config/common.py`
- 离线建库配置：`rag/config/indexing.py`
- 在线查询配置：`rag/config/retrieval.py`

当前共有：

- 13 个命令行参数：预处理 6 个、图片说明与切分导出 4 个、离线建库 2 个、在线查询 1 个
- 50 个唯一环境变量，其中 `RAG_DEPENDENCY_VECTOR_ENABLED` 同时影响离线建库和在线查询

## 1. 参数生效方式与优先级

### 1.1 四个命令的配置来源

| 命令 | 命令行参数 | 环境变量 | 关键行为 |
| --- | --- | --- | --- |
| `python -m rag.cli.preprocess_docs` | 是 | 否 | 输出目录从 CLI 读取；相对路径按项目根目录解析 |
| `python -m rag.cli.prepare_chunks` | 是 | 仅切分尺寸和可选存储目录 | 不调用标注或向量接口。省略 `--storage-dir` 时使用项目根目录下的 `storage` |
| `python -m rag.cli.build_index` | 是 | 是 | 读取已切分 chunk。显式 `--chunks`、`--storage-dir` 覆盖默认位置；相对路径按项目根目录解析 |
| `python -m rag.cli.answer_query` | 仅问题文本 | 是 | 查询配置全部从环境变量读取 |

### 1.2 覆盖优先级

离线建库通过命令行运行时：

1. 显式 `--storage-dir` 覆盖 `RAG_STORAGE_DIR`
2. 显式 `--chunks` 覆盖默认的 `storage/prepared_chunks/chunks.jsonl`
3. 两者都省略时，存储目录使用 `RAG_STORAGE_DIR`，未设置则使用项目根目录下的 `storage`
4. 其他建库配置从环境变量读取
5. 未设置的环境变量使用代码默认值

相对路径一律按项目根目录解析，不使用启动命令时的工作目录。因此在 `rag/cli` 下运行，也不会把索引写到 `rag/cli/storage`。

在线查询通过命令行运行时：

1. 位置参数 `question` 决定问题文本
2. 所有服务、模型、路径和召回参数从环境变量读取
3. 未设置的环境变量使用代码默认值

通过 Python API 调用时：

- `build_offline_index(config=...)` 中显式传入的 `config` 优先于环境变量
- `build_offline_index(docs_dir=..., storage_dir=...)` 中显式路径优先于 `config.paths`
- `build_query_service(config=...)` 中显式传入的 `config` 优先于环境变量

## 2. 通用解析规则

### 2.1 路径

相对路径按执行命令时的当前工作目录解析。建议始终在仓库根目录执行命令。

### 2.2 整数与浮点数

整数参数使用 `int(...)` 解析，浮点参数使用 `float(...)` 解析。传入非数字文本会在配置加载阶段抛出 `ValueError`，项目没有自动纠正机制。

当前代码没有对全部数值做统一范围校验。除非某个参数在本文中另有说明，数量、半径、批大小和字符数应使用正数；比例和置信度应使用 `0.0` 到 `1.0` 之间的值。

### 2.3 布尔值

布尔环境变量忽略大小写和首尾空格。以下值会被解析为 `true`：

```text
1, true, yes, on
```

环境变量未设置时使用各自默认值。除上述四种真值外，其他任何已设置值都会被解析为 `false`，包括 `0`、`false`、`no`、`off`、空字符串和拼写错误。

## 3. 文档预处理 CLI 参数

命令入口：

```powershell
.venv/Scripts/python.exe -m rag.cli.preprocess_docs [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input-dir` | 路径 | 项目 `storage/wiki-export` | 原始 `.doc`、`.docx` 文档目录。目录不存在或没有支持的文件时直接失败 |
| `--html-dir` | 路径 | 项目 `storage/converted_html` | LibreOffice 转换后的中间 HTML 输出目录。默认是项目根目录下的绝对路径 |
| `--markdown-dir` | 路径 | 项目 `storage/cleaned_markdown` | 清洗后的 Markdown 输出目录，同时写入 `source_manifest.json` |
| `--image-dir` | 路径 | 项目 `storage/cleaned_assets` | 从文档或 MHTML 中提取的图片资产目录 |
| `--soffice-path` | 路径或空 | `None` | 显式指定 `soffice.exe`；未指定时自动搜索系统命令和常见安装目录 |
| `--no-recursive` | 开关 | `false` | 指定后只扫描输入目录一级文件；默认递归扫描所有子目录 |

`--soffice-path` 未设置时按以下顺序查找：

1. `PATH` 中的 `soffice`
2. `PATH` 中的 `libreoffice`
3. `C:\Program Files\LibreOffice\program\soffice.exe`
4. `C:\Program Files (x86)\LibreOffice\program\soffice.exe`

示例：

```powershell
.venv/Scripts/python.exe -m rag.cli.preprocess_docs `
  --input-dir "D:/ai/project/rag/storage/wiki-export"
```

## 3.1 图片说明与切分导出 CLI 参数

命令入口：

```powershell
.venv/Scripts/python.exe -m rag.cli.prepare_chunks [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input-dir` | 路径 | 项目 `storage/wiki-export` | 原始 `.doc`、`.docx` 目录，传给与 `preprocess_docs` 相同的预处理函数。默认是项目根目录下的绝对路径 |
| `--storage-dir` | 路径 | 项目根目录下的 `storage` | 预处理、图片说明、增强 Markdown 和 chunk 导出的根目录。默认是绝对路径；显式传入的相对路径按项目根目录解析 |
| `--captions` | 路径 | `<storage-dir>/image_captions.json` | 图片中文说明。键是 Markdown 中的图片相对路径 |
| `--skip-preprocess` | 开关 | `false` | 跳过预处理，直接使用已有 `cleaned_markdown` |

切分尺寸不单独提供命令行参数，读取 `RAG_CHUNK_SIZE` 和 `RAG_CHUNK_OVERLAP`，默认分别是 `512` 和 `100`。这两个值只在 `prepare_chunks` 切分时生效。改完后需要重新导出 chunk，`build_index` 才会看到新的边界。

输出固定在 `--storage-dir` 下：

- `cleaned_markdown/`、`converted_html/`、`cleaned_assets/`
- `enriched_markdown/`
- `prepared_chunks/chunks.jsonl`
- `prepared_chunks/report.json`

示例：

```powershell
.venv/Scripts/python.exe -m rag.cli.prepare_chunks --skip-preprocess
```

## 4. 离线建库 CLI 参数

命令入口：

```powershell
.venv/Scripts/python.exe -m rag.cli.build_index [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--chunks` | 路径 | `<storage>/prepared_chunks/chunks.jsonl` | `prepare_chunks` 导出的 chunk 文件。建库不再扫描 Markdown，也不再切分 |
| `--storage-dir` | 路径 | 项目根目录下的 `storage` | 索引工作区、发布清单和最终索引根目录。显式传入时覆盖 `RAG_STORAGE_DIR`；相对路径按项目根目录解析 |

示例：

```powershell
.venv/Scripts/python.exe -m rag.cli.build_index
```

指定 chunk 文件时：

```powershell
.venv/Scripts/python.exe -m rag.cli.build_index `
  --chunks "storage/prepared_chunks/chunks.jsonl" `
  --storage-dir "storage"
```

## 5. 在线查询 CLI 参数

命令入口：

```powershell
.venv/Scripts/python.exe -m rag.cli.answer_query [question]
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `question` | 可选位置参数 | `我要接入iOS订阅，现在需要如何处理` | 用户问题；传入空白字符串会在查询服务中报“问题不能为空” |

示例：

```powershell
.venv/Scripts/python.exe -m rag.cli.answer_query "我要接入谷歌订阅，现在需要如何处理"
```

## 6. 共享路径与服务环境变量

### 6.1 路径配置

| 环境变量 | 类型 | 默认值 | 使用阶段 | 说明 |
| --- | --- | --- | --- | --- |
| `RAG_DOCS_DIR` | 路径 | 项目 `storage/enriched_markdown`，目录不存在时用同级 `cleaned_markdown` | 在线读取原文，建库读取整篇文档 | 供 `read_chunk`、精确查找，以及建库时为片段写定位上下文。建库不按这个目录重新切分 |
| `RAG_STORAGE_DIR` | 路径 | 项目根目录下的 `storage` | 预处理、离线建库、在线查询 | 相对路径按项目根目录解析。查询时用于读取根 `manifest.json` |

### 6.2 LLM 服务配置

| 环境变量 | 类型 | 默认值 | 使用阶段 | 说明 |
| --- | --- | --- | --- | --- |
| `DASHSCOPE_API_KEY` | 字符串或空 | 无，实际为 `None` | 离线标注、依赖图、查询侧 LLM | 离线建库实际必需；查询侧不设置时进入无 LLM 降级模式 |
| `DASHSCOPE_BASE_URL` | URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 所有 LLM 调用 | OpenAI 兼容 Chat Completions 接口地址 |

查询侧未设置 `DASHSCOPE_API_KEY` 时：

- Query Rewrite 不创建，不执行改写召回
- LLM 精排不创建，只保留规则粗排
- 依赖路由仍执行确定性规则，规则未命中时不调用 LLM
- 答案生成退回“最相关资料摘要”文本

查询侧已设置 Key 但 LLM 请求失败时，各组件行为并不完全相同：

| 组件 | 请求异常后的实际行为 |
| --- | --- |
| Query Rewrite | 异常继续向上抛出，本次查询失败 |
| 依赖召回路由 | 捕获异常，保守关闭本次依赖扩展 |
| LLM 精排 | 捕获异常，退回规则粗排 |
| 最终答案生成 | 异常继续向上抛出，本次查询失败 |

### 6.3 Embedding 服务配置

| 环境变量 | 类型 | 默认值 | 使用阶段 | 说明 |
| --- | --- | --- | --- | --- |
| `RAG_EMBEDDING_API_KEY` | 字符串或空 | 无，实际为 `None` | 离线建库、在线查询 | 当前实现不会回退到 `DASHSCOPE_API_KEY`，需要单独设置 |
| `RAG_EMBEDDING_BASE_URL` | URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 离线建库、在线查询 | 当前实现不会读取 `DASHSCOPE_BASE_URL` 作为回退值 |
| `RAG_EMBEDDING_MODEL` | 字符串 | `text-embedding-v4` | 离线建库、在线查询 | 查询模型必须与已发布索引清单中的模型名称完全一致 |
| `RAG_EMBEDDING_BATCH_SIZE` | 整数 | `20` | 离线建库 | checkpoint 外层批大小；底层兼容客户端仍会按最多 `10` 条拆分请求 |

重要说明：虽然 `load_embedding_config_from_env()` 接收 LLM 配置参数，当前实现并未使用该参数。因此，只设置 `DASHSCOPE_API_KEY` 不会为 Embedding 提供 Key，只设置 `DASHSCOPE_BASE_URL` 也不会改变 Embedding 地址。

离线建库在向量化前会显式检查 `RAG_EMBEDDING_API_KEY`。在线查询初始化时没有等价的空 Key 校验，但第一次请求查询向量时仍会因凭据不可用而失败。因此该 Key 对正常建库和查询都属于实际必需配置。

## 7. 离线切分与语义标注环境变量

### 7.1 Chunk 切分

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_CHUNK_SIZE` | 整数 | `512` | `prepare_chunks` 使用的目标 token 大小；切分会尽量保持 Markdown 表格、代码块和语义块完整 |
| `RAG_CHUNK_OVERLAP` | 整数 | `100` | `prepare_chunks` 中相邻 chunk 复用的目标 token 大小 |

建议满足 `RAG_CHUNK_SIZE > 0` 且 `0 <= RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE`。当前代码没有集中校验这组关系，不合理值可能产生异常或不符合预期的切分结果。

### 7.2 Chunk 语义标注

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_ANNOTATOR_MODEL` | 字符串 | `qwen3.6-plus` | 离线 chunk 语义标注模型 |
| `RAG_ANNOTATION_WORKERS` | 整数 | `5` | 并发标注线程数，应大于 `0` |
| `RAG_ANNOTATION_PROMPT_PATH` | 路径 | `prompts/annotation_v4.md` | 标注提示词文件；文件不存在时使用代码内置提示词 |
| `RAG_ANNOTATION_PROMPT_VERSION` | 字符串 | `annotation_v4` | 标注输出协议版本；与历史状态不一致时触发全量语义重建 |

`RAG_ANNOTATION_PROMPT_VERSION` 表示输出协议兼容性，不是普通文案版本。仅修改示例或措辞时不应随意递增，否则会强制重建全部文档。

### 7.2.1 片段定位上下文

定位上下文在标注之前生成。标注、正文向量和 BM25 读取同一份检索文本。摘要、关键词和标签使用定位说明；依赖主题只依据正文。

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_CHUNK_CONTEXT_MODEL` | 字符串 | 无，必须设置 | 为每个 chunk 写定位上下文的模型。未设置时建库直接失败，不补默认模型 |
| `RAG_CHUNK_CONTEXT_WORKERS` | 整数 | `4` | 并发生成定位上下文的线程数 |
| `RAG_CHUNK_CONTEXT_PROMPT_PATH` | 路径 | `prompts/chunk_context.md` | 定位提示词。相对路径按项目根目录解析，必须包含 `{{WHOLE_DOCUMENT}}` 和 `{{CHUNK_CONTENT}}` |
| `RAG_CHUNK_CONTEXT_PROMPT_VERSION` | 字符串 | `chunk_context_v1` | 定位提示词协议版本。变化时重新标注、重新嵌入并重建 BM25 |

### 7.3 文档依赖图

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_DEPENDENCY_GRAPH_MODEL` | 字符串 | `qwen3.6-plus` | 离线文档级依赖图聚合模型 |
| `RAG_DEPENDENCY_GRAPH_PROMPT_PATH` | 路径 | `prompts/dependency_graph_v3.md` | 依赖图提示词；文件不存在时使用代码内置提示词 |
| `RAG_DEPENDENCY_GRAPH_PROMPT_VERSION` | 字符串 | `dependency_graph_v3` | 依赖图输出协议版本；与历史状态不一致时触发全量语义重建 |
| `RAG_DEPENDENCY_VECTOR_ENABLED` | 布尔 | `false` | 建库时是否创建 `dependency_vec` 向量集合 |

`RAG_DEPENDENCY_VECTOR_ENABLED` 同时被在线查询读取。要使用依赖向量兜底，建库和查询必须都设置为 `true`，并重新构建索引。查询开启但索引中没有 `dependency_vec` 时会记录警告并关闭依赖向量兜底，不影响普通依赖图扩展。

## 8. 在线多路召回环境变量

### 8.1 原始问题召回

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_RETRIEVAL_CONTENT_TOP_K` | 整数 | `8` | 正文向量 `content_vec` 的召回条数 |
| `RAG_RETRIEVAL_SUMMARY_TOP_K` | 整数 | `6` | 摘要向量 `summary_vec` 的召回条数 |
| `RAG_RETRIEVAL_BM25_TOP_K` | 整数 | `8` | BM25 关键词召回条数 |

三个通道的结果会按 `node_id` 聚合，并由规则 RRF 粗排。这里的 Top K 是各路召回上限，不是最终答案引用数。

### 8.2 Query Rewrite

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_RETRIEVAL_REWRITE_ENABLED` | 布尔 | `true` | 是否启用查询改写；还要求存在 `DASHSCOPE_API_KEY` |
| `RAG_QUERY_REWRITE_MODEL` | 字符串 | `qwen3.6-plus` | 查询改写模型 |
| `RAG_RETRIEVAL_REWRITE_LIMIT` | 整数 | `3` | 最多保留的改写查询数量；模型提示词要求生成 2 到 4 条，但最终按该值截断 |
| `RAG_RETRIEVAL_REWRITE_CONTENT_TOP_K` | 整数 | `4` | 每条改写问题的正文向量召回数 |
| `RAG_RETRIEVAL_REWRITE_SUMMARY_TOP_K` | 整数 | `3` | 每条改写问题的摘要向量召回数 |
| `RAG_RETRIEVAL_REWRITE_BM25_TOP_K` | 整数 | `4` | 每条改写问题的 BM25 召回数 |

改写召回的理论原始命中上限为：

```text
改写数量 x (rewrite_content_top_k + rewrite_summary_top_k + rewrite_bm25_top_k)
```

实际候选数通常更少，因为不同通道和不同改写可能命中同一 `node_id`。

### 8.3 邻居扩展

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_RETRIEVAL_NEIGHBOR_ENABLED` | 布尔 | `true` | 是否补充同文档相邻 chunk |
| `RAG_RETRIEVAL_NEIGHBOR_RADIUS` | 整数 | `1` | 向中心 chunk 前后扩展的 chunk 距离 |
| `RAG_RETRIEVAL_CENTER_TOP_K` | 整数 | `5` | 选取规则粗排前多少个候选作为邻居扩展中心 |

半径 `1` 表示尝试加入每个中心 chunk 的前一个和后一个 chunk。邻居候选在规则分中有固定降权，避免仅因位置相邻超过直接召回结果。

## 9. 在线依赖召回环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_DEPENDENCY_RECALL_ENABLED` | 布尔 | `true` | 是否启用依赖路由和一层依赖图扩展 |
| `RAG_DEPENDENCY_ROUTER_LLM_ENABLED` | 布尔 | `true` | 规则未命中时是否允许 LLM 判断是否需要依赖召回 |
| `RAG_DEPENDENCY_ROUTER_MODEL` | 字符串 | `qwen3.6-plus` | 依赖召回路由模型 |
| `RAG_DEPENDENCY_SEED_SCORE_RATIO` | 浮点数 | `0.7` | seed 候选相对最高规则分的最低比例，建议范围 `0.0` 到 `1.0` |
| `RAG_DEPENDENCY_MAX_SEEDS` | 整数 | `8` | 最多选取的 seed 数量 |
| `RAG_DEPENDENCY_MIN_CONFIDENCE` | 浮点数 | `0.7` | 允许扩展的依赖边最低置信度，建议范围 `0.0` 到 `1.0` |
| `RAG_DEPENDENCY_PER_SEED_LIMIT` | 整数 | `5` | 每个 seed 最多补充的依赖节点数 |
| `RAG_DEPENDENCY_TOTAL_LIMIT` | 整数 | `12` | 最终组装的 seed 与依赖证据总数上限 |
| `RAG_DEPENDENCY_VECTOR_ENABLED` | 布尔 | `false` | 查询时是否尝试加载并查询 `dependency_vec` |

依赖扩展只读取一层直接依赖，不递归遍历整张图。`RAG_DEPENDENCY_TOTAL_LIMIT` 在启用依赖扩展时还会约束最终传给答案生成器的上下文数量，因此其语义比普通召回 Top K 更接近“依赖场景最终上下文预算”。

即使没有 LLM Key，依赖路由仍会先执行关键词规则：流程、接入、配置、排查类问题可能开启依赖扩展；字段含义、错误码等事实查询会关闭依赖扩展。

## 10. 在线重排环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_RERANK_LLM_ENABLED` | 布尔 | `true` | 是否对规则粗排头部候选执行 LLM 精排；还要求存在 `DASHSCOPE_API_KEY` |
| `RAG_RERANK_LLM_MODEL` | 字符串 | `qwen3.6-plus` | LLM 精排模型 |
| `RAG_RERANK_LLM_TOP_N` | 整数 | `10` | 送入 LLM 精排的规则粗排候选数 |
| `RAG_RERANK_FINAL_TOP_N` | 整数 | `5` | 未启用依赖扩展时，最终传给答案生成器的候选数量 |

规则粗排器内部固定使用 `rrf_k=60`，当前没有对应环境变量。LLM 精排后的融合权重固定为规则分 `20%`、LLM 分 `80%`，当前也没有对应环境变量。

当依赖扩展实际发生时，最终上下文按依赖关系重新组装，主要受 `RAG_DEPENDENCY_TOTAL_LIMIT` 控制，不再简单使用 `RAG_RERANK_FINAL_TOP_N` 截断。

## 11. 答案生成环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_ANSWER_ENABLED` | 布尔 | `true` | 是否调用 LLM 生成最终答案 |
| `RAG_ANSWER_MODEL` | 字符串 | `qwen3.6-plus` | 最终答案生成模型 |
| `RAG_ANSWER_CONTEXT_TOP_K` | 整数 | `4` | 兼容保留参数；当前实现不会用它二次截断候选列表 |
| `RAG_ANSWER_MAX_CONTEXT_CHARS` | 整数 | `1200` | 每个候选写入答案提示词时最多保留的字符数 |

`RAG_ANSWER_CONTEXT_TOP_K` 当前只被保存到 `AnswerGenerator` 对象，没有参与 `generate()` 的上下文切片。调整它不会改变实际引用数量。实际数量由普通场景的 `RAG_RERANK_FINAL_TOP_N` 或依赖场景的 `RAG_DEPENDENCY_TOTAL_LIMIT` 决定。

当 `RAG_ANSWER_ENABLED=false` 或没有 `DASHSCOPE_API_KEY` 时，不会返回空答案，而是输出候选摘要或正文片段，并保留引用来源。

## 12. 必需配置与降级矩阵

| 场景 | 必需配置 | 缺失后的行为 |
| --- | --- | --- |
| 文档预处理 | 可用的 LibreOffice `soffice` | 无法转换文档，直接失败 |
| 离线语义标注 | `DASHSCOPE_API_KEY` | `SemanticAnnotator` 初始化失败 |
| 离线向量化 | `RAG_EMBEDDING_API_KEY` | 建库在向量化前直接失败 |
| 在线向量召回 | `RAG_EMBEDDING_API_KEY` | 查询向量请求失败，无法执行检索 |
| Query Rewrite | `DASHSCOPE_API_KEY` | 未设置时自动关闭；已设置但请求失败时整次查询失败 |
| 依赖路由 LLM | `DASHSCOPE_API_KEY` | 退回确定性规则；路由请求失败时保守关闭依赖扩展 |
| LLM 精排 | `DASHSCOPE_API_KEY` | 退回规则粗排 |
| LLM 答案生成 | `DASHSCOPE_API_KEY` | 未设置时输出资料摘要；已设置但请求失败时整次查询失败 |

## 13. 索引一致性要求

已发布索引的根 `manifest.json` 会记录建库使用的 `embedding_model`。在线查询初始化时严格比较当前 `RAG_EMBEDDING_MODEL` 与清单值：

- 名称完全一致：允许查询
- 名称不一致：初始化失败
- 旧清单缺少 `embedding_model`：初始化失败

切换 Embedding 模型后必须重新构建并发布索引，不能直接复用旧向量库。

以下参数发生变化时也应考虑重建：

| 参数 | 是否需要重建 | 原因 |
| --- | --- | --- |
| `RAG_CHUNK_SIZE`、`RAG_CHUNK_OVERLAP` | 是 | 先重新执行 `prepare_chunks`。chunk 边界和 `node_id` 变化后，`build_index` 会替换受影响的 chunk |
| `RAG_EMBEDDING_MODEL` | 是 | 向量空间变化，且查询有严格一致性校验 |
| `RAG_ANNOTATION_PROMPT_VERSION` | 自动全量重建 | 代码按版本变化使历史标注失效 |
| `RAG_CHUNK_CONTEXT_PROMPT_VERSION` | 自动重新标注并嵌入 | 定位说明变化后，摘要、正文向量和 BM25 一起更新 |
| `RAG_DEPENDENCY_GRAPH_PROMPT_VERSION` | 自动全量重建 | 代码按版本变化使历史依赖图失效 |
| `RAG_DEPENDENCY_VECTOR_ENABLED: false -> true` | 是 | 需要生成 `dependency_vec` 集合 |
| 纯在线召回、重排、答案参数 | 否 | 不改变持久化索引内容 |

## 14. PowerShell 配置示例

### 14.1 最小完整配置

```powershell
$env:DASHSCOPE_API_KEY = "你的 LLM Key"
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:RAG_EMBEDDING_API_KEY = "你的 Embedding Key"
$env:RAG_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:RAG_EMBEDDING_MODEL = "text-embedding-v4"
```

### 14.2 调整建库参数

```powershell
$env:RAG_CHUNK_SIZE = "768"
$env:RAG_CHUNK_OVERLAP = "120"
$env:RAG_ANNOTATION_WORKERS = "8"
$env:RAG_EMBEDDING_BATCH_SIZE = "20"

.venv/Scripts/python.exe -m rag.cli.build_index
```

### 14.3 关闭查询侧 LLM 增强，只保留基础召回和规则排序

```powershell
$env:RAG_RETRIEVAL_REWRITE_ENABLED = "false"
$env:RAG_DEPENDENCY_ROUTER_LLM_ENABLED = "false"
$env:RAG_RERANK_LLM_ENABLED = "false"
$env:RAG_ANSWER_ENABLED = "false"
```

这不会关闭 Embedding。在线检索仍必须配置 `RAG_EMBEDDING_API_KEY`。

### 14.4 提高召回范围

```powershell
$env:RAG_RETRIEVAL_CONTENT_TOP_K = "12"
$env:RAG_RETRIEVAL_SUMMARY_TOP_K = "10"
$env:RAG_RETRIEVAL_BM25_TOP_K = "12"
$env:RAG_RETRIEVAL_REWRITE_LIMIT = "3"
$env:RAG_RERANK_LLM_TOP_N = "15"
$env:RAG_RERANK_FINAL_TOP_N = "8"
```

提高这些值会增加候选数量、LLM 输入和查询耗时，并不保证答案质量单调提升。

### 14.5 开启依赖向量

```powershell
$env:RAG_DEPENDENCY_VECTOR_ENABLED = "true"

.venv/Scripts/python.exe -m rag.cli.build_index
```

建库完成后，查询进程也必须保留同一环境变量值。

## 15. 主要输出位置

下面的 `storage/` 都是项目根目录下的 `storage/`，不是运行脚本时所在目录。

| 阶段 | 默认输出 |
| --- | --- |
| 预处理中间 HTML | `storage/converted_html/` |
| 清洗后 Markdown | `storage/cleaned_markdown/` |
| 图片资产 | `storage/cleaned_assets/` |
| 来源映射清单 | `storage/cleaned_markdown/source_manifest.json` |
| 带图片说明的 Markdown | `storage/enriched_markdown/` |
| 切分导出 | `storage/prepared_chunks/chunks.jsonl` |
| 评测题 | `storage/eval/questions.jsonl` |
| 固定建库工作区 | `storage/index_workspace/` |
| 根发布清单 | `storage/manifest.json` |
| 最终在线索引 | `storage/<final_index_dir>/` |

建库失败时工作区会保留用于排查，但根 `manifest.json` 不会切换，在线查询继续使用上一份已成功发布的完整索引。

## 16. 当前实现中的易错点

1. `RAG_EMBEDDING_BATCH_SIZE` 的代码默认值是 `20`，底层单次请求上限是 `10`，两者不是同一个概念。
2. Embedding Key 和 Base URL 当前不会从 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 回退。
3. 省略 `--storage-dir` 时使用 `RAG_STORAGE_DIR`；环境变量也未设置时，使用项目根目录下的 `storage`。`RAG_DOCS_DIR` 在建库时用来读取整篇文档并写定位上下文，不参与切分。
4. `RAG_ANSWER_CONTEXT_TOP_K` 当前不参与实际上下文截断。
5. 布尔值拼写错误不会报错，而会被静默当作 `false`。
6. Query Rewrite 和答案生成的远程调用异常不会自动降级，会使本次查询失败。
7. 数值参数缺少统一范围校验，建议不要传入 `0`、负数或不合理比例。
