# RAG 逻辑审查报告

审查日期：2026-11-19

审查范围：检索、排序、依赖扩展、答案生成、切分、BM25 入库和 Agent 工具。结论来自阅读代码，并用 `storage/prepared_chunks/chunks.jsonl`（235 个 chunk）上的脚本复现验证。当前环境未安装 pytest，`tests/` 未运行。

按影响从大到小排列。

## 一、已确认的缺陷（真实数据能复现）

### 1. 切分时会丢掉超长的原子章节

位置：`rag/indexing/markdown_chunker.py:275-280`、`322-327`

- 当原子 H3 章节超过 1000 token（`H3_ATOMIC_MAX_TOKENS`）时，`_regular_chunking_append` 把块追加进 `current_blocks`，但调用方随即执行 `current_blocks = []`，换成了一个新列表。块存进了旧列表，旧列表又被丢弃，所以整个章节一个字都不会入库。
- 真实数据：`国内支付网关-接口信息.md` 丢了 40 行，对应 `### ~~查询订单~~` 这一节（1308 token）。这一节恰好是已废弃的接口，所以目前影响不大。但以后任何大接口表都会被静默丢掉。

### 2. heading_path 层级错误

位置：`rag/indexing/markdown_chunker.py:140`、`238`

- `heading_stack[: level - 1]` 默认栈的下标等于标题级别减 1。但每个 H2 分组里没有 H1，栈就整体错位，平级的 H3 被当成了父子关系。
- 真实数据：
  - `接口信息 > 生成预支付订单 > 查询用户订单列表`
  - `Apple订阅服务接入 > 1、邮件申请 > 2、实时通知`
- 另外，一个 chunk 覆盖整个 H2 章节时，路径取的是最后一个子标题，而不是 H2 本身。
- heading_path 会进入正文向量、BM25、LLM 精排和答案提示，所以这个错误会影响整条链路。
- 修法：栈里存 `(level, title)`，遇到新标题时弹出所有级别大于或等于它的项；文档的 H1 要带进每个分组。

### 3. 只有标题的碎片 chunk

235 个 chunk 里有 22 个不到 40 字，比如 `### 内购订单支付接口`、`# 支付接入和权限申请指引`。它们会占用召回名额，却没有信息量。

### 4. BM25 的词频全部失效

位置：`rag/retrieval/tokenization.py:34`、`rag/indexing/storage_indexer.py:260`

- `tokenize_technical_text` 会对 token 去重，而 BM25 建库用的也是它，所以每个词在文档里的 TF 恒为 1。BM25 退化成了按 IDF 加权的"出现或不出现"匹配。
- 分词也没有过滤停用词和全角标点。例如"谷歌订阅的回调通知怎么配置？"会切出 `的 / 怎么 / 什么 / ？ / （ / ）` 这类 token。
- 修法：给 BM25 单独写一个分词函数，不去重，并过滤停用词和标点。只需要重建 `bm25.pkl`，不用重新嵌入。

### 5. BM25 会返回 0 分的命中

位置：`rag/retrieval/retriever.py:225`

无论分数多少都取前 top_k 条。和查询完全无关的 chunk 也会进入 RRF，还能拿到多路命中奖励。应该过滤掉 `score <= 0` 的结果。

## 二、打分尺度不一致

### 6. 粗排的加分项压过了 RRF

位置：`rag/retrieval/ranking.py:33-47`

- 单条命中的 RRF 分最多是 1/61 ≈ 0.016。但每多一路命中加 0.05，coherence 加减 0.03，相邻扩展的 chunk 减 0.06。结果排序主要由命中了几路和 coherence 决定，召回排名本身影响很小。
- content_vec 和 summary_vec 用的是同一个嵌入模型，结果高度相关，却算作两路去拿奖励。
- 建议把这些加分项缩到和 RRF 同一量级，或者改成按路由加权的 RRF。

### 7. 依赖 seed 的选择基本由标题关键词决定

位置：`rag/retrieval/dependency_recall.py:199`

- `final_score`（约 0.02 到 0.15）乘 0.7，`capability_score`（0 到 1）乘 0.3，后者实际占主导。所以 seed 按标题里有没有"配置/签名"这类词来选，和问题的相关度关系不大。
- `assemble_dependency_context` 还会把这些 seed 排在常规 TopN 前面（第 352 行）。
- 建议先把 final_score 归一化（min-max 或按名次），再和 capability_score 混合。
- `quality_score` 这条规则（第 173 行）是死代码：标注器从不产出这个字段，所以它永远取默认值 1.0。

### 8. LLM 精排的输入和输出都需要约束

位置：`rag/retrieval/ranking.py:120-148`

- 提示词没有规定分数范围和评分标准，返回的分数也没有被限制到 [0, 1]。
- 每个候选只给模型看前 400 字，而 74% 的 chunk（175/235）都超过 400 字。

## 三、健壮性

### 9. 查询改写和答案生成都没有异常兜底

- `QueryRewriter.rewrite`（`rag/retrieval/ranking.py:78-106`）：模型只要返回了非 JSON 内容，`json.loads` 就会抛异常，整个查询失败。
- `AnswerGenerator.generate`（`rag/retrieval/answer_generator.py:62`）：接口超时同样会让整个查询失败。
- 精排和依赖路由出错时都会降级，这两处没有，行为不一致。

### 10. 答案上下文按单个 chunk 截断，且没有总量上限

- 每个 chunk 被截到 1200 字（`max_context_chars`），33% 的 chunk（77/235）会被截断，大接口表首当其冲。
- 依赖扩展开启时，上下文最多可能有 seed（≤8）+ TopN（8）+ 强制依赖（15），约 31 个 chunk，却没有总长度预算。
- 建议改成按总字数分配上下文，不再截单个 chunk。
- `context_top_k` 已经没有任何作用。文档写明这是有意去掉的截断，但配置项还留着。

## 四、性能

### 11. 一次查询里的 LLM 调用全是串行的

- 当前顺序是：改写 LLM → 原问题和 3 条改写各调一次嵌入接口 → 依赖路由 LLM → 精排 LLM → 答案 LLM。
- 依赖路由只依赖原问题，可以和改写同时跑。
- 改写后的查询可以用已有的 `get_text_embedding_batch` 一次嵌入完。
- `find_dependency_fallback`（`rag/retrieval/retriever.py:119-136`）每处理一条缺失的依赖边，就要把同文档的所有 chunk 重新分词一遍。现在语料小，优先级低。

## 五、Agent 工具

### 12. 几个工具的小问题

- `get_dependencies`（`rag/agent/tools.py:165`）没有用配置里的 `min_confidence`，也不检查目标 node 是否存在。主流程会做同文档兜底，这里没有，所以可能返回一个 `read_chunk` 读不到的 node_id。
- 工具调用超过 20 轮时直接返回固定文本（`rag/agent/tools.py:715`）。更好的做法是最后再调一次模型，设 `tool_choice="none"`，让它根据已有资料作答。
- `grep_docs` 区分大小写（`rag/agent/tools.py:460`），`getUser` 搜不到 `getuser`。

## 建议的修复顺序

1. 先修第 1、2、4、5 项，它们直接影响召回质量。
   - 注意：修第 1、2 项会改变 chunk 指纹，受影响的文档要重新标注和嵌入，会产生 LLM 调用费用。
   - 第 4、5 项只需要重建 BM25。
2. 再修第 9 项（异常兜底）和第 6、7 项（打分尺度）。
3. 其余按需处理。
