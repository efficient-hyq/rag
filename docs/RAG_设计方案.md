# RAG 系统完整设计方案

> 技术栈：LlamaIndex + Qwen DashScope API（LLM + Embedding）+ Chroma  
> 适用场景：本地测试环境，混合格式文档（md / pdf / html）  
> 版本：v1.3

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [阶段划分](#2-阶段划分)
3. [模块一：文档加载](#3-模块一文档加载)
4. [模块二：规则切分](#4-模块二规则切分)
5. [模块三：LLM 语义标注](#5-模块三llm-语义标注)
6. [模块四：Embedding 向量化](#6-模块四embedding-向量化)
7. [模块五：多路入库](#7-模块五多路入库)
8. [模块六：多路召回](#8-模块六多路召回)
9. [模块七：LLM Rerank](#9-模块七llm-rerank)
10. [模块八：答案生成](#10-模块八答案生成)
11. [技术选型汇总](#11-技术选型汇总)
12. [数据结构设计](#12-数据结构设计)
13. [成本控制策略](#13-成本控制策略)
14. [工程目录结构](#14-工程目录结构)
15. [关键设计原则与风险](#15-关键设计原则与风险)

---

## 1. 整体架构概览

系统分为两条主线：**离线索引流水线**（一次性执行）和**在线查询流水线**（每次问答触发）。

```
┌─────────────────────────────────────────────────────────────┐
│                     离线索引流水线                            │
│                                                             │
│  原始文档  →  文档加载  →  规则切分  →  LLM语义标注           │
│  (md/pdf/html)                         ↓                    │
│                              Embedding向量化                 │
│                                   ↓                         │
│                ┌──────────────────┬──────────────────┐      │
│                ▼                  ▼                  ▼      │
│           向量库(Chroma)      BM25关键词索引      元数据JSON  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     在线查询流水线                            │
│                                                             │
│  用户Query  →  三路并行召回  →  合并去重  →  LLM Rerank       │
│                                              ↓              │
│                                       答案生成(qwen3.6-plus) │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段划分

| 阶段 | 类型 | 触发时机 | 主要 API 消耗 |
|------|------|----------|--------------|
| 文档加载 | 离线 | 新文档接入时 | 无 |
| 规则切分 | 离线 | 新文档接入时 | 无 |
| LLM 语义标注 | 离线 | 新 chunk 产生时 | qwen-turbo（DashScope） |
| Embedding 向量化 | 离线 | 新 chunk 产生时 | text-embedding-v2（DashScope，免费额度） |
| 多路入库 | 离线 | 新 chunk 产生时 | 无 |
| 多路召回 | 在线 | 每次查询 | text-embedding-v2（DashScope，1次） |
| LLM Rerank | 在线 | 每次查询 | qwen-turbo（1次） |
| 答案生成 | 在线 | 每次查询 | qwen3.6-plus（1次） |

---

## 3. 模块一：文档加载

### 3.1 职责

将磁盘上不同格式的原始文档统一转换为 LlamaIndex `Document` 对象，保留来源元数据。

### 3.2 技术选型

使用 **LlamaIndex `SimpleDirectoryReader`** 作为统一入口，按文件扩展名分发不同的 Reader：

| 格式 | Reader | 说明 |
|------|--------|------|
| `.md` | `FlatReader` | 保留原始 Markdown 文本，不做额外解析 |
| `.pdf` | `PDFReader` | 按页拆分，保留页码信息 |
| `.html` | `HTMLTagReader` | 提取 `<body>` 正文，过滤导航/广告标签 |

### 3.3 关键设计决策

- 开启 `filename_as_id=True`，用文件路径作为 `doc_id`，便于后续增量更新时精确定位
- 开启 `recursive=True`，支持多层子目录
- PDF 使用 `return_full_document=False`，按页切分而非整文档，避免单个 Document 过大

### 3.4 输出结构

每个 `Document` 携带以下 metadata：

```
doc_id        : 文件路径（唯一标识）
file_name     : 文件名
file_type     : md / pdf / html
page_label    : 页码（PDF 专有）
creation_date : 文件创建时间
```

---

## 4. 模块二：规则切分

### 4.1 职责

将 Document 按语义边界切成大小均匀的 chunk（TextNode），同时保留 overlap 解决跨 chunk 语义断裂问题。

### 4.2 技术选型

使用 **LlamaIndex `SentenceSplitter`**，优于 `TokenTextSplitter` 的原因：

- 以句子为最小切分单元，不在句子中间断开
- 天然保留段落语义完整性
- 支持中文断句（依赖 `jieba` 或内置规则）

### 4.3 核心参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 token | 平衡语义完整性与 embedding 质量 |
| `chunk_overlap` | 100 token | 约 20%，保证跨 chunk 上下文连续 |
| `paragraph_separator` | `\n\n` | 段落为优先切分点 |

### 4.4 切分优先级

```
Markdown 标题（# ## ###）> 段落（\n\n）> 句子（。！？）> 空格
```

超过 `chunk_size` 时按上述优先级从高到低选择切分点，确保不在词语中间断裂。

### 4.5 输出结构

每个 `TextNode` 在原有 metadata 基础上追加：

```
chunk_index  : 在文档内的顺序编号（0, 1, 2...）
token_size   : 实际 token 数
node_id      : UUID（全局唯一）
```

---

## 5. 模块三：LLM 语义标注

### 5.1 职责

这是系统的**核心增强层**。不修改 chunk 内容，而是为每个 chunk 生成多维度的语义元数据，作为后续多路检索的入口。

### 5.2 技术选型

- 模型：**qwen-turbo**（成本最低，标注任务不需要复杂推理）
- 接口：DashScope OpenAI 兼容模式（`base_url` 切换即可复用 openai SDK）
- 并发：`ThreadPoolExecutor`，控制并发数 ≤ 5，避免触发限速

### 5.3 标注字段定义

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `summary` | string | 不超过 50 字的核心内容概括 | summary 向量召回的检索入口 |
| `keywords` | list[str] | 3~6 个核心关键词 | BM25 召回增强 |
| `tags` | list[str] | 领域/主题标签（如"支付"、"API文档"） | 过滤筛选 |
| `type` | enum | text / api / code / table | 差异化处理策略 |
| `has_code` | bool | 是否包含代码块 | 代码类 chunk 特殊处理 |
| `coherence` | enum | high / medium / low | 标记语义不完整的 chunk |

### 5.4 Prompt 设计原则

- 明确禁止 LLM 修改原文、拆分或合并 chunk
- 要求严格输出 JSON，不附带任何解释文字
- 对输出做容错处理：去除 markdown 代码块包裹、指数退避重试（最多 3 次）
- 兜底策略：全部重试失败时写入空结构，不阻塞流程

### 5.5 并发与限速策略

```
并发数      : 5（可配置）
失败重试    : 最多 3 次，指数退避（1s → 2s → 4s）
触发 429 时 : 将并发数降至 2，等待 60s 后重试
批次检查点  : 每完成 20 个 chunk 打印进度
```

### 5.6 增量策略

标注结果写回 `TextNode.metadata`，并持久化至 `metadata.json`。增量新增文档时，仅对新 chunk 做标注，不重跑已有 chunk。

---

## 6. 模块四：Embedding 向量化

### 6.1 职责

将 chunk 文本转为向量，支持后续语义相似度检索。采用**双路向量**策略提升召回质量。通过 DashScope API 调用，使用免费额度覆盖本地测试阶段全部消耗。

### 6.2 技术选型

| 维度 | 选型 | 说明 |
|------|------|------|
| 模型 | `text-embedding-v2` | DashScope 成熟 Embedding 模型，中英文效果均衡 |
| 向量维度 | 固定 1536 维 | v2 不支持自定义维度，Chroma 集合初始化时对应设置 |
| 集成方式 | `openai` SDK（OpenAI 兼容接口） | 与标注模型调用方式一致，便于统一网关、Key 和重试治理 |
| 接口 | OpenAI Embeddings 兼容接口 | 可直连 DashScope 兼容模式，也可接入第三方中转站 |

### 6.3 OpenAI 兼容接口说明

向量化不再使用 LlamaIndex 官方 DashScope Embedding 集成包，而是和语义标注一样通过 `openai.OpenAI(api_key, base_url)` 创建客户端，再调用 `client.embeddings.create(...)`。

核心参数：
```
model    : RAG_EMBEDDING_MODEL，默认 text-embedding-v4
api_key  : RAG_EMBEDDING_API_KEY，未配置时回退到 RAG_LLM_API_KEY / DASHSCOPE_API_KEY
base_url : RAG_EMBEDDING_BASE_URL，未配置时回退到 RAG_LLM_BASE_URL / DASHSCOPE_BASE_URL
```

> **注意**：OpenAI 兼容 Embeddings 接口没有 LlamaIndex DashScope 集成包里的 `text_type` 参数。索引和查询阶段需要使用同一个 embedding 模型与兼容接口，确保向量空间一致。

### 6.4 双路向量策略

| 路 | 向量化对象 | 存储集合 | 检索偏重 |
|----|-----------|---------|---------|
| 路 1 | `node.text`（原文） | `content_vec`（1536 维） | 精确语义匹配 |
| 路 2 | `node.metadata.summary`（LLM 生成的摘要） | `summary_vec`（1536 维） | 主题级宏观匹配 |

summary 向量的优势：摘要经过 LLM 提炼，去除噪音词，与用户 query 的语义空间更接近，能召回原文向量遗漏的相关 chunk。

### 6.5 批量调用策略

- 通过 OpenAI 兼容 Embeddings 接口批量传入文本列表
- 离线阶段对全量 chunk 批量调用，在线查询阶段仅对单条 query 调用一次
- 查询阶段的单次 embedding 调用约消耗 50 token，免费额度几乎不受影响

---

## 7. 模块五：多路入库

### 7.1 存储架构

系统使用三个互补的存储结构，分别服务不同的召回路径：

```
storage/
├── chroma/           # 向量数据库（Chroma 本地持久化）
│   ├── content_vec   # content embedding 集合
│   └── summary_vec   # summary embedding 集合
├── bm25.pkl          # BM25 关键词索引（pickle 序列化）
└── metadata.json     # 完整 chunk 结构（原文 + 全量 metadata）
```

### 7.2 各存储选型说明

**Chroma（向量库）**
- 纯 Python，零配置，`PersistentClient` 数据落本地文件夹
- 支持 `upsert`，天然支持增量写入
- 与 LlamaIndex 原生集成，后续可无缝切换至 Qdrant

**rank-bm25（关键词索引）**
- 纯 Python，无任何依赖
- 支持中文（分词后输入）
- 序列化为 pickle 文件，重启后直接加载，不需要重建

**metadata.json（元数据存储）**
- 存储每个 chunk 的完整信息（原文 + 所有 metadata）
- Rerank 和答案生成阶段取原文时的数据源
- 结构：`{node_id: {text, summary, keywords, tags, ...}}`

### 7.3 入库流程

```
1. 计算所有 chunk 的 content embedding（批量）
2. 计算所有 chunk 的 summary embedding（批量）
3. Chroma upsert content_vec 集合
4. Chroma upsert summary_vec 集合
5. 构建 BM25 索引，pickle 序列化落盘
6. 构建 metadata dict，JSON 序列化落盘
```

---

## 8. 模块六：多路召回

### 8.1 职责

从三个不同维度同时检索候选 chunk，合并去重后形成候选集，交给 Rerank 精排。

### 8.2 三路召回详情

| 路 | 方式 | 数据源 | TopK | 优势 |
|----|------|--------|------|------|
| 路 1 | content 向量召回 | Chroma content_vec | 8 | 精确语义相似度 |
| 路 2 | summary 向量召回 | Chroma summary_vec | 6 | 主题级宏观相关 |
| 路 3 | BM25 关键词召回 | rank-bm25 | 8 | 精确关键词命中，补充向量遗漏 |

三路合并后候选集约 **15~22 个 chunk**（有重叠），进入 Rerank 精排。

### 8.3 合并去重策略

按召回顺序合并，先出现的 chunk 优先级更高（体现在去重后的排序中）：

```
优先级：路1（content向量）> 路2（summary向量）> 路3（BM25）
```

同一 `node_id` 只保留第一次出现，后续重复丢弃。

### 8.4 Query 向量化

查询阶段仅调用一次 embedding 接口，生成的 query 向量同时用于路 1 和路 2 的向量检索，不重复计算。

---

## 9. 模块七：LLM Rerank

### 9.1 职责

对候选集进行精排，解决向量召回"语义相似但答案无关"的问题，是系统最终质量的关键保障。

### 9.2 技术选型

- 模型：**qwen-turbo**（成本控制，rerank 属于判断题，不需要生成能力）
- 策略：单次调用批量打分（一次 LLM 调用处理所有候选），而非逐个调用

### 9.3 打分维度

LLM 对每个候选 chunk 输出：

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | 0~10 | 与 query 的相关性分数 |
| `reason` | string | 简短评分理由（调试用） |

输入上下文包含：query 原文 + chunk 正文（截断至 400 字）+ title_path（如有）

### 9.4 容错策略

- JSON 解析失败时：按原顺序返回 top-N，不阻塞流程
- 最终保留 **Top 3~5** 个 chunk 进入答案生成

---

## 10. 模块八：答案生成

### 10.1 职责

将 Top chunks 拼接为上下文，调用 LLM 生成最终答案，并附带来源引用。

### 10.2 技术选型

- 模型：**qwen3.6-plus**（生成质量优先，此阶段每次查询只调用一次）

### 10.3 上下文拼接策略

每个 chunk 在上下文中附带来源标注：

```
[来源 1：xxx.md  chunk-12]
...chunk 正文...

---

[来源 2：yyy.pdf  第3页  chunk-45]
...chunk 正文...
```

### 10.4 Prompt 设计原则

- 明确要求"基于参考资料回答，不要超出资料范围"
- 无相关信息时输出"根据现有文档无法回答"，避免幻觉
- 要求在答案末尾注明来源（文件名 + chunk 序号）
- `temperature=0.3`，偏保守，减少随机生成

---

## 11. 技术选型汇总

| 层次 | 组件 | 选型 | 理由 |
|------|------|------|------|
| 框架编排 | RAG 框架 | LlamaIndex | RAG 专项封装，Node/Document 体系完整 |
| 文档加载 | 多格式 Reader | LlamaIndex SimpleDirectoryReader | 内置多格式支持，自动路由 |
| 文本切分 | Splitter | LlamaIndex SentenceSplitter | 句子边界切分，语义完整 |
| 语义标注 | LLM | Qwen qwen-turbo | 成本最低，批量任务 |
| 向量化 | Embedding 模型 | text-embedding-v4（OpenAI 兼容接口） | 与标注模型统一调用方式，便于接入 DashScope 兼容模式或第三方中转站 |
| 向量存储 | 向量库 | Chroma（本地持久化） | 零配置，纯Python，LlamaIndex原生支持 |
| 关键词检索 | BM25 | rank-bm25 | 纯Python，无依赖，够用 |
| 元数据存储 | KV存储 | JSON文件 | 本地测试够用，读写简单 |
| Rerank | LLM | Qwen qwen-turbo | 成本低，判断题场景 |
| 答案生成 | LLM | Qwen qwen3.6-plus | 生成质量优先 |
| LLM 接口 | SDK | openai（兼容模式）| DashScope 兼容 OpenAI 格式 |
| Embedding 接口 | SDK | openai（兼容模式）| 与 LLM 标注共用 OpenAI 兼容调用方式 |

---

## 12. 数据结构设计

### 12.1 TextNode 完整结构（标注后）

```json
{
  "node_id": "uuid-xxxx",
  "text": "...chunk原文内容...",
  "metadata": {
    "doc_id": "./docs/payment.md",
    "file_name": "payment.md",
    "file_type": "md",
    "chunk_index": 12,
    "token_size": 487,
    "summary": "介绍订阅支付的创建流程及关键参数",
    "keywords": ["订阅", "支付", "创建", "参数"],
    "tags": ["支付系统", "API文档"],
    "type": "api",
    "has_code": true,
    "coherence": "high"
  }
}
```

### 12.2 metadata.json 结构

```json
{
  "uuid-xxxx": {
    "text": "...chunk原文...",
    "doc_id": "./docs/payment.md",
    "file_name": "payment.md",
    "chunk_index": 12,
    "summary": "...",
    "keywords": ["..."],
    "tags": ["..."],
    "type": "api",
    "has_code": true,
    "coherence": "high"
  }
}
```

### 12.3 召回结果结构

```json
{
  "id": "uuid-xxxx",
  "text": "...chunk原文...",
  "metadata": { "...": "..." },
  "source": "content_vec | summary_vec | bm25"
}
```

---

## 13. 成本控制策略

### 13.1 调用分层

```
【云端 DashScope API】
  qwen-turbo          → 语义标注（离线，批量，一次性）          付费
  qwen-turbo          → LLM Rerank（在线，每次查询 1 次）       付费
  qwen3.6-plus        → 答案生成（在线，每次查询 1 次）         付费
  text-embedding-v2   → Embedding（离线批量 + 在线每次 1 次）   免费额度覆盖

【本地运行，完全免费】
  rank-bm25   → BM25 关键词检索
  Chroma      → 向量存储与检索
```

### 13.2 text-embedding-v2 免费额度说明

DashScope 对新用户提供 Embedding 免费调用额度，本地测试阶段的全量索引和日常查询通常可被免费额度完全覆盖。以下是消耗估算供参考：

| 场景 | Token 消耗（参考） |
|------|-----------------|
| 1000 个 chunk 离线 content embedding | 约 50 万 token |
| 1000 个 chunk 离线 summary embedding | 约 5 万 token |
| 每次在线查询（query embedding） | 约 50 token |

### 13.3 LLM 离线阶段成本估算

以 1000 个 chunk 为例（每 chunk 约 500 token）：

| 步骤 | 模型 | 调用量 | 预估费用（参考） |
|------|------|--------|----------------|
| 语义标注 | qwen-turbo | 1000 次，约 60 万 token | 约 ¥0.6 |
| content embedding | text-embedding-v2 | 约 50 万 token | 免费额度 |
| summary embedding | text-embedding-v2 | 约 5 万 token | 免费额度 |
| **合计** | | | **约 ¥0.6** |

### 13.4 LLM 在线阶段成本估算（单次查询）

| 步骤 | 模型 | Token 消耗 | 预估费用 |
|------|------|-----------|---------|
| Query embedding | text-embedding-v2 | ~50 token | 免费额度 |
| Rerank | qwen-turbo | ~5000 token | 约 ¥0.005 |
| 答案生成 | qwen3.6-plus | ~3000 token | 约 ¥0.01 |
| **合计** | | | **约 ¥0.015/次** |

### 13.5 其他节省策略

- 标注失败的 chunk 用兜底空结构，不重试浪费费用
- 增量更新只标注新 chunk，历史 chunk 不重跑
- summary 字段控制在 50 字以内，embedding token 消耗极小
- 离线索引一次性完成后，后续查询仅消耗少量 LLM token

---

## 14. 工程目录结构

```
rag_project/
├── rag/
│   ├── cli/
│   │   ├── preprocess_docs.py      # 文档预处理入口
│   │   ├── build_index.py          # 离线建库入口
│   │   └── answer_query.py         # 在线查询入口
│   ├── preprocess/
│   │   └── document_preprocessor.py
│   ├── indexing/
│   │   ├── document_loader.py
│   │   ├── markdown_chunker.py
│   │   ├── semantic_annotator.py
│   │   ├── embedding_client.py
│   │   ├── storage_indexer.py
│   │   ├── preview_renderer.py
│   │   └── index_builder.py
│   ├── retrieval/
│   │   ├── tokenization.py
│   │   ├── retriever.py
│   │   ├── ranking.py
│   │   ├── answer_generator.py
│   │   └── query_service.py
│   ├── shared/
│   │   ├── checkpoints.py
│   │   └── logging_utils.py
│   └── config/
├── prompts/
├── docs/
├── tests/
├── storage/
└── requirements.txt
```

### 14.1 依赖清单

```
llama-index
llama-index-readers-file
openai                             # DashScope/OpenAI 兼容模式，用于标注与向量化
chromadb
rank-bm25
pymupdf
beautifulsoup4
lxml
```

---

## 15. 关键设计原则与风险

### 15.1 核心设计原则

**原则一：结构稳定，不允许 LLM 修改 chunk**
LLM 只做语义标注，输出写入 metadata，不修改 `node.text`。chunk 的边界由规则切分严格决定，保证系统行为可预期。

**原则二：多路召回是质量下限保障**
单路向量召回存在语义偏差盲区（query 与 chunk 用词差异大时召回失败）。三路互补确保只要有一路命中，候选集就包含正确答案。

**原则三：summary 是第二语义入口**
原文 embedding 受噪音词干扰，summary embedding 语义更纯净，对"问题与答案表述差异大"的场景尤为重要。

**原则四：Rerank 是质量上限保障**
向量召回保"不漏"，Rerank 保"不错"。两者缺一不可。

### 15.2 已知风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| DashScope 限速（429） | 标注阶段中断 | 并发数≤5，指数退避重试，断点续标 |
| LLM 标注输出非法 JSON | 标注失败 | 正则清洗 + 兜底空结构 + 3次重试 |
| PDF 解析乱码 | chunk 质量差 | 使用 pymupdf，OCR 类 PDF 需额外处理 |
| BM25 中文分词粗糙 | 关键词召回精度低 | 可引入 jieba 分词增强，低优先级 |
| chunk 跨语义边界 | 检索结果不完整 | overlap 100 token 缓解，coherence 字段标记低质 chunk |
| Chroma 大数据量性能下降 | 召回延迟增加 | 本地测试 ≤10万 chunk 无问题，生产环境换 Qdrant |

### 15.3 后续可扩展方向

- **Query Rewrite**：用 LLM 将用户 query 改写为多个检索子句，提升召回率 30~50%
- **Chunk 动态扩展**：Rerank 后将 top chunk 的前后相邻 chunk 一并纳入上下文，解决答案跨 chunk 问题
- **混合权重融合**：对三路召回结果做加权融合（RRF 算法），优于简单去重合并
- **存储升级**：本地测试完成后，Chroma 可无缝替换为 Qdrant（Docker），JSON 可替换为 SQLite

---

*文档结束*
