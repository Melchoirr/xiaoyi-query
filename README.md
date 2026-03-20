# 小易时序 RAG 预测微服务

基于向量检索（RAG）和逆距离加权（IDW）融合算法的时序预测微服务。运行时无需任何外部服务，数据完全存储在本地磁盘。

---

## 目录

- [技术栈](#技术栈)
- [架构概览](#架构概览)
- [项目结构](#项目结构)
- [全局配置](#全局配置)
- [Layer 1 — 底层检索引擎](#layer-1--底层检索引擎)
- [Layer 2 — Agentic Router](#layer-2--agentic-router)
- [Layer 3 — 精排融合引擎](#layer-3--精排融合引擎)
- [Foundation Model — TS2Vec](#foundation-model--ts2vec)
- [API 接口](#api-接口)
- [安装与启动](#安装与启动)
- [数据摄入](#数据摄入)
- [Layer 3 训练](#layer-3-训练)
- [预测与测试](#预测与测试)
- [注意事项](#注意事项)

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步 API + 自动文档 |
| 向量数据库 | Qdrant（本地磁盘模式） | 无外部依赖 |
| 特征编码器 | TS2Vec（ONNX Runtime） | 预训练时序基础模型 |
| 融合排序 | IDW + XGBoost 微调 | 逆距离加权融合 |
| 语义理解 | OpenAI GPT-4o-mini | 意图解析 + 报告生成 |
| 数据处理 | NumPy + Pandas | Z-Score 归一化、滑动窗口 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Layer 2: Agentic Router                          │
│                         POST /api/v2/agent/forecast                     │
│                                                                          │
│   user_query ──► TimeRAGAgent.parse_intent() ──► Qdrant Filter JSON     │
│                           │                                              │
│   history_x ──► TSProcessor.normalize() ──► (normalized, μ, σ)          │
│                           │                                              │
│               ONNXEncoder.encode(TS2Vec ONNX) ──► query_vector (320-dim)│
│                           │                                              │
│               QdrantRetriever.search(query_vector, Filter) ──► Top-K   │
│                           │                                              │
│               FusionRanker.rank_and_fuse(Top-K, query_meta) ──► y_hat (精排融合预测)   │
│                           │                                              │
│               TimeRAGAgent.generate_report(y_hat, chunks) ──► AI 报告   │
│                           │                                              │
│                    ┌──────▼──────┐                                      │
│                    │ JSON Response │                                     │
│                    │ prediction_values  (数值预测)                       │
│                    │ ai_analysis_report (文字解析)                       │
│                    └──────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Layer 1: 底层检索引擎                             │
│                                                                          │
│   scripts/ingest_ett.py                                                 │
│       ├── TSProcessor.normalize(history_x)                                │
│       ├── ONNXEncoder.encode(TS2Vec ONNX) ──► 320-dim vector           │
│       └── QdrantRetriever.ingest_batch() ──► Qdrant (payload + time feat)│
│                                                                          │
│   Qdrant Collection "time_series_rag"                                    │
│       └── 每个点: {vector: [320], payload: {mu, sigma, future_y,        │
│                                      month, hour, is_weekend,           │
│                                      time_of_day, source}}              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
xiaoyi-query/
├── models/                              # 预训练模型
│   ├── foundation_encoder.onnx         # TS2Vec Encoder ONNX（由 scripts/export_foundation_model.py 导出）
│   └── xgb_ranker.json                 # XGBoost 精排模型（由 scripts/train_ranker.py 训练）
├── core/
│   ├── processor.py                    # TSProcessor - Z-Score 归一化/反归一化
│   └── config.py                       # 全局配置中心（LLM Provider、路径、系统常量）
├── components/
│   ├── encoder.py                      # ONNXEncoder - TS2Vec 向量化引擎
│   ├── retriever.py                   # QdrantRetriever - 向量存储与过滤检索
│   ├── ranker.py                      # FusionRanker - Layer 3 XGBoost 精排 + Softmax 融合
│   └── agent.py                       # TimeRAGAgent - LLM 意图解析 + 报告生成
├── api/
│   ├── schemas.py                     # Layer 1 Pydantic 数据模型
│   ├── schemas_agent.py                # Layer 2 Pydantic 数据模型
│   └── routes_agent.py                # Layer 2 Agentic FastAPI 路由
├── scripts/
│   ├── ingest_ett.py                  # ETT 数据批量摄入（含时间特征）
│   ├── export_foundation_model.py      # TS2Vec ONNX 导出工具（离线运行）
│   └── train_ranker.py                # XGBoost 精排模型离线训练（离线运行）
├── ETT_data/                           # ETT 数据集（ETTh1, ETTh2, ETTm1, ETTm2）
├── qdrant_data/                        # Qdrant 本地存储（gitignored）
├── .env                                 # 环境变量配置（API Keys，见 .env.example）
├── main.py                              # FastAPI 入口 + 依赖注入
└── README.md
```

---

## 全局配置

所有系统常量、路径、LLM Provider 配置集中在 `core/config.py`。

### 配置方式（优先级从高到低）

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | 环境变量 | 生产环境动态注入，如 `OPENAI_API_KEY=sk-xxx` |
| 2 | `.env` 文件 | 本地开发使用，参考 `.env.example` |
| 3 | `core/config.py` | 硬编码默认值 |

### `.env` 示例文件（参考）

```bash
# LLM Provider 配置（支持 openai / deepseek / qwen / ollama）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
# DEEPSEEK_API_KEY=sk-xxx
# QWEN_API_KEY=sk-xxx

# 可选：自定义 Base URL（用于代理或本地服务）
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 支持的 LLM Provider

| Provider | Base URL | 默认模型 | 说明 |
|----------|----------|----------|------|
| `openai` | `api.openai.com/v1` | `gpt-4o-mini` | OpenAI 官方 API |
| `deepseek` | `api.deepseek.com/v1` | `deepseek-chat` | DeepSeek 官方 API |
| `qwen` | `dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | 通义千问 API |
| `ollama` | `localhost:11434/v1` | `llama3.2` | 本地 Ollama 服务 |

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `system.input_length` | 100 | 历史窗口长度 |
| `system.default_future_length` | 48 | 默认预测窗口长度 |
| `system.qdrant_collection` | `time_series_rag` | Qdrant 集合名 |
| `system.onnx_encoder_path` | `models/foundation_encoder.onnx` | TS2Vec ONNX 路径 |
| `system.xgb_ranker_path` | `models/xgb_ranker.json` | XGBoost 精排模型路径 |
| `llm.provider` | `openai` | LLM Provider |
| `llm.model` | `gpt-4o-mini` | 模型名称 |

---

## Layer 1 — 底层检索引擎

### 1.1 TSProcessor（core/processor.py）

实现实例级 Z-Score 归一化，即每个序列独立计算均值 μ 和标准差 σ：

- `normalize(sequence)` — 计算 μ, σ，返回归一化数组及统计量
- `denormalize(sequence, mu, sigma)` — 将预测结果还原为真实量级

### 1.2 ONNXEncoder（components/encoder.py）

使用预训练时序基础模型 **TS2Vec** 的 ONNX 编码器（详见 [Foundation Model](#foundation-model--ts2vec)）：

- `encode(sequence)` — 输入长度为 100 的归一化数组，输出 **320 维** L2 归一化嵌入向量
- 内部流程：Zero-padding → ONNX 推理 → Mean Pooling → L2 Normalize

### 1.3 QdrantRetriever（components/retriever.py）

- 使用 `qdrant_client.QdrantClient(path="./qdrant_data")` 纯本地存储
- `create_collection_if_not_exists()` — 创建使用 **Cosine** 距离的 collection（向量维度 320）
- `ingest_batch()` — 批量写入，payload 包含数值特征 + 时间特征
- `search(query_vector, top_k, query_filter)` — 向量检索 + Payload 过滤，返回 Top-K

**Payload Schema：**

```json
{
  "source":     "OT",
  "future_y":   [18.3, 19.1, ...],
  "mu":         25.4,
  "sigma":      3.2,
  "history_x":  [30.1, 30.5, ...],
  "month":      7,
  "hour":       20,
  "is_weekend": true,
  "time_of_day": "evening"
}
```

### 1.4 FusionRanker（components/ranker.py）

接收检索返回的 Top-K 结果集（包含 score 和对应的 future_y），分两阶段融合：

1. **IDW 融合**（主阶段）— 根据相似度分数计算权重，加权求和得到初步预测
2. **XGBoost 微调**（辅助阶段）— 用 `[distance, mu, sigma]` 构建特征，对 IDW 结果做微调修正

权重计算公式：\( w_i = \frac{1}{d_i^2 + \epsilon} \)

最终预测：\( \hat{y} = \sum_i \frac{w_i}{\sum_j w_j} \cdot future\_y_i \)

---

## Layer 2 — Agentic Router

### 2.1 TimeRAGAgent（components/agent.py）

**职责 A — 意图解析 `parse_intent(query: str) -> dict`：**

调用 LLM（`gpt-4o-mini`），将用户的自然语言查询严格映射为 Qdrant Filter 条件 JSON。

示例：
- 输入：`"结合当前夏季周末的用电高峰，预测未来趋势"`
- 输出：
```json
{
  "must": [
    {"key": "is_weekend",  "match": {"value": true}},
    {"key": "month",       "range": {"gte": 6, "lt": 9}}
  ],
  "should": [],
  "must_not": []
}
```

**职责 B — 报告生成 `generate_report(query, prediction, chunks) -> str`：**

调用 LLM，结合预测数值和召回片段元数据，生成结构化 Markdown 分析报告，包含：
- 查询意图解读
- 预测结果统计摘要
- 历史相似样本分析
- 专业建议

### 2.2 编排链路（POST /api/v2/agent/forecast）

```
Step 1: TimeRAGAgent.parse_intent(user_query)     → Qdrant Filter JSON
Step 2: TSProcessor.normalize(history_x)            → (normalized, μ, σ)
Step 3: ONNXEncoder.encode(normalized)              → query_vector (320-dim)
Step 4: QdrantRetriever.search(query_vector, Filter)→ Top-K 片段
Step 5: FusionRanker.rank_and_fuse(Top-K, query_metadata)  → y_hat (反归一化)
Step 6: Payload 反归一化                            → 原始量级 future_y
Step 7: TimeRAGAgent.generate_report(...)          → AI 分析报告
Step 8: 返回 JSON {prediction_values, ai_analysis_report, ...}
```

---

## Layer 3 — 精排融合引擎

### 3.1 特征工程（训练 / 推理完全对齐）

`scripts/train_ranker.py` 中的 `_build_cross_features()` 和 `components/ranker.py` 中的同名函数**逻辑完全一致**，通过共享的 `extract_time_features()` 保证特征口径统一。

| 序号 | 特征名 | 公式 | 含义 |
|------|--------|------|------|
| 0 | `distance` | `1 - cosine_score` | 向量距离 |
| 1 | `abs_mu_diff` | `|μ_query - μ_doc|` | 均值差异 |
| 2 | `abs_sigma_diff` | `|σ_query - σ_doc|` | 标准差差异 |
| 3 | `is_same_month` | `1 if doc.month == query.month else 0` | 月份相同 |
| 4 | `is_same_time_of_day` | `1 if doc.tod == query.tod else 0` | 时段相同 |
| 5 | `is_same_weekday` | `1 if doc.is_weekend == query.is_weekend else 0` | 工作日/周末相同 |
| 6 | `mu_ratio` | `μ_doc / (μ_query + 1e-6)` | 均值比率 |

### 3.2 训练流程（离线）

```
ETT 数据集
    ↓ 随机采样 N 个时间点
每个采样点:
    history_x → TSProcessor.normalize() → (normalized, μ_q, σ_q)
    → ONNXEncoder.encode() → query_vector
    → QdrantRetriever.search() → Top-K 召回片段
    对每个召回片段:
        反归一化 future_y_k → 原始量级
        Pearson(future_y_k, true_future) → 质量分数标签 y ∈ [-1, 1]
        _build_cross_features() → 7 维特征 X
    → XGBoost.fit(X, y) → models/xgb_ranker.json
```

### 3.3 在线推理（FusionRanker）

```
Qdrant 召回 Top-K 片段 (K 条)
    ↓
_build_cross_features(query_metadata, doc_payload, distance) × K
    ↓
XGBoost.predict(X) → raw_scores ∈ ℝ^K（每个片段的原始质量分）
    ↓
Softmax(raw_scores) → weights ∈ (0,1)^K，sum(weights) = 1
    ↓
weighted_sum(future_y_k, weights) → y_hat ∈ ℝ^future_length
```

**降级策略**：若 `models/xgb_ranker.json` 不存在，自动降级为 IDW 融合，并在 stderr 打印警告。

---

## Foundation Model — TS2Vec

### 模型选型依据

我们选择 **TS2Vec（Time Series to Vector）** 作为特征编码器，原因如下：

1. **学术背景**：TS2Vec（Yue et al., 2022）是时序表征学习的里程碑工作，在 UCR/UEA/TSRR 等多个基准上取得了 SOTA 的 Zero-shot 表征迁移性能。

2. **架构适合 RAG 检索**：对时间维度做 Mean Pooling 后得到全局序列向量，非常适合作为语义检索的 Query/Document Embedding。

3. **动态输入长度 + 固定输出维度**：接受任意长度序列，输出固定 320 维稠密向量，兼容 Qdrant 的固定向量维度要求。

4. **纯 PyTorch，无自定义算子**：可直接导出为 ONNX，onnxruntime 可原生推理，无特殊依赖。

5. **跨域 Zero-shot 能力**：在 128 个数据集上预训练，涵盖医疗、能源、传感器等多元领域，ETT 数据无需微调即可直接使用。

### ONNX 输入 / 输出规格

| | 形状 | 说明 |
|---|---|---|
| **输入** `[batch=1, seq_len=512, channels=1]` | `float32` | TS2Vec 固定序列长度 |
| **输出** `[batch=1, seq_len=512, embedding_dim=320]` | `float32` | 每个时间步的上下文表征 |

> 注意：低于 512 的序列在**头部**补零（pre-padding，因果卷积无泄漏）；超过 512 的序列截取最后 512 个点。

### 导出模型（离线）

```bash
# 在有 torch + ts2vec 环境的独立机器上运行：
pip install torch ts2vec onnxscript

python scripts/export_foundation_model.py \
    --output-dir models \
    --output-name foundation_encoder.onnx \
    --input-length 512 \
    --device cpu

# 将 models/foundation_encoder.onnx 拷贝到目标机器
```

---

## API 接口

### Layer 1 接口

#### 健康检查

```
GET /health
```

#### 统计信息

```
GET /stats
```

#### 摄入数据

```
POST /ingest
Content-Type: application/json

{
  "history_x": [1.0, 2.0, ...],   // 长度必须为 100
  "future_y": [6.0, 7.0, ...]    // 任意长度
}
```

#### 预测（数值链路）

```
POST /predict
Content-Type: application/json

{
  "history_x": [5.0, 6.0, ...],   // 长度必须为 100
  "top_k": 5
}
```

### Layer 2 接口（Agentic）

#### Agentic 预测（完整链路）

```
POST /api/v2/agent/forecast
Content-Type: application/json

{
  "user_query": "结合当前夏季周末的用电高峰，预测未来趋势",
  "history_x": [30.1, 30.5, ...],   // 长度必须为 100
  "top_k": 10
}
```

**响应：**

```json
{
  "success": true,
  "intent_filter": {
    "must": [
      {"key": "is_weekend",  "match": {"value": true}},
      {"key": "month",       "range": {"gte": 6, "lt": 9}}
    ]
  },
  "prediction_values": [24.3, 25.1, 26.0, ...],   // 数值预测（始终返回）
  "retrieved_chunks": [
    {
      "id": 1234,
      "score": 0.952,
      "future_y": [24.1, 24.9, ...],
      "mu": 30.2,
      "sigma": 2.8,
      "time_features": {
        "month": 7,
        "hour": 20,
        "is_weekend": true,
        "time_of_day": "evening"
      },
      "source": "OT"
    }
  ],
  "ai_analysis_report": "## 1. 查询意图解读\n...",
  "message": "基于 Top-10 相似片段（含过滤）预测，共召回 10 条"
}
```

#### 独立意图解析

```
POST /api/v2/agent/intent
Content-Type: application/json

{
  "user_query": "周末晚上高油温会怎么走"
}
```

---

## 安装与启动

### 第一步：创建虚拟环境（首次使用）

```bash
# 在项目根目录下创建 .venv
python -m venv .venv

# 激活虚拟环境（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 激活虚拟环境（Windows Git Bash / WSL）
source .venv/Scripts/activate
```

### 第二步：安装依赖

```bash
# 安装所有依赖包
.venv\Scripts\python.exe -m pip install -e .

# 或使用 uv（已在项目根目录配置 pyproject.toml）
uv sync
```

### 第三步：导出 Foundation Model（如尚未导出）

> TS2Vec ONNX 模型是整个系统的特征编码器，**必须导出后才能启动服务**。

```bash
# 在独立 torch 环境中运行（推荐 GPU，CPU 亦可）：
python scripts/export_foundation_model.py --device cpu

# 成功后将生成 models/foundation_encoder.onnx
```

### 第四步：启动 FastAPI 服务

```bash
# 使用 .venv 中的 python 启动（确保使用正确的依赖环境）
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

> 服务启动时自动加载 `models/foundation_encoder.onnx`，并在 Qdrant 中自动创建 collection。

### 第五步：摄入时序数据

> **注意**：摄入时请先**停止 FastAPI 服务**（Qdrant 本地模式不支持并发写入）。

先预览摄入计划（Dry Run）：

```bash
python scripts/ingest_ett.py --file ETTh1 --column OT --dry-run
```

确认无误后执行摄入：

```bash
# 摄入指定列
python scripts/ingest_ett.py --file ETTh1 --column OT

# 摄入全部列（耗时约 5~10 分钟）
python scripts/ingest_ett.py --file ETTh1
```

### 第六步：训练精排模型（可选，推荐）

> Layer 3 XGBoost 精排模型可显著提升检索质量。**离线运行，不需要启动 FastAPI 服务**。

```bash
# 默认参数：5000 个采样点，Top-K=20
python scripts/train_ranker.py

# 自定义参数
python scripts/train_ranker.py --num-samples 10000 --top-k 20

# 强制重新训练（覆盖已有模型）
python scripts/train_ranker.py --force
```

### 第七步：访问服务并测试

服务启动后，访问以下地址：

- Swagger UI：http://127.0.0.1:8000/docs
- ReDoc：http://127.0.0.1:8000/redoc

---

## 数据摄入

> **注意**：摄入时请先**停止 FastAPI 服务**（Qdrant 本地模式不支持并发写入）。

### 预览摄入计划（Dry Run）

不写入数据库，仅预览摄入计划：

```bash
python scripts/ingest_ett.py --file ETTh1 --column OT --dry-run
```

### 执行摄入

```bash
# 摄入指定列（推荐）
python scripts/ingest_ett.py --file ETTh1 --column OT

# 摄入全部列（耗时约 5~10 分钟）
python scripts/ingest_ett.py --file ETTh1
```

**摄入参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history-len` | 100 | 历史窗口长度 |
| `--future-len` | 48 | 预测窗口长度 |
| `--batch-size` | 500 | 每批写入 Qdrant 的条数 |

**摄入量估算（`--history-len 100 --future-len 48`）：**

| 数据集 | 原始行数 | 可生成样本数 |
|--------|----------|-------------|
| ETTm1 (单列 OT) | 69,653 | ~69,506 |
| ETTm1 (全部7列) | 69,653 | ~486,542 |
| ETTh1 (单列 OT) | 17,420 | ~17,273 |

> **重要**：升级到 TS2Vec 后，**必须重新摄入数据**（清空 `qdrant_data/` 目录后重新摄入），因为向量维度从 128 变更为 320。

---

## Layer 3 训练

> **前置条件**：Qdrant 中已摄入数据（`qdrant_data/` 非空）且 ONNX 编码器已导出。

### 训练命令

```bash
# 默认参数（5000 采样点，Top-K=20）
python scripts/train_ranker.py

# 自定义参数
python scripts/train_ranker.py --file ETTh1 --column OT --num-samples 10000 --top-k 20

# 强制重新训练（覆盖已有模型）
python scripts/train_ranker.py --force
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--file` | ETTh1 | ETT 数据集文件名 |
| `--column` | OT | 要采样的数值列名 |
| `--num-samples` | 5000 | 采样数量 |
| `--top-k` | 20 | Qdrant 检索的 Top-K |
| `--history-len` | 100 | 历史窗口长度 |
| `--future-len` | 48 | 未来窗口长度 |

**前置条件：**
- Qdrant 中已摄入数据（`qdrant_data/` 非空）
- ONNX 编码器模型已存在（`models/foundation_encoder.onnx` 或 `encoder_v1.onnx`）

**训练输出：**

```
models/xgb_ranker.json          # XGBoost 精排模型
models/xgb_ranker.meta.json     # 训练元信息（特征重要性、验证集 RMSE/R² 等）
```

**训练完成后：重启 FastAPI 服务**即可自动加载新模型。

---

## 预测与测试

### 1. 检查向量数量

```bash
curl http://127.0.0.1:8000/health
```

响应示例：

```json
{
  "status": "healthy",
  "collection_name": "time_series_rag",
  "vector_count": 69506,
  "embedding_dim": 320
}
```

### 2. Layer 1 数值预测测试

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history_x": [30.0, 30.5, 31.0, ...],  // 100个数值
    "top_k": 5
  }'
```

### 3. Layer 2 Agentic 预测测试

```bash
curl -X POST http://127.0.0.1:8000/api/v2/agent/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "夏季周末用电高峰时，未来48小时的走势如何？",
    "history_x": [30.0, 30.5, 31.0, ...],  // 100个数值
    "top_k": 10
  }'
```

> 需要配置 `OPENAI_API_KEY` 环境变量才能使用 Layer 2 接口。

### 4. 从 ETT 数据中提取真实样本测试

```python
import pandas as pd
import requests

df = pd.read_csv("ETT_data/ETTm1.csv")
column = "OT"
history_len = 100
future_len = 48

history_x = df[column].iloc[1000:1100].tolist()
true_future = df[column].iloc[1100:1148].tolist()

response = requests.post("http://127.0.0.1:8000/predict", json={
    "history_x": history_x,
    "top_k": 5
})

prediction = response.json()["prediction"]

import numpy as np
mae = np.mean(np.abs(np.array(prediction) - np.array(true_future)))
rmse = np.sqrt(np.mean((np.array(prediction) - np.array(true_future)) ** 2))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```

---

## 注意事项

- `history_x` 长度必须为 **100**（由 `INPUT_LENGTH` 配置）
- 所有数据存储在本地 `./qdrant_data` 目录，删除该目录将清除所有已摄入数据
- `models/foundation_encoder.onnx` 需通过 `scripts/export_foundation_model.py` 导出，删除后需重新导出
- **升级 TS2Vec 后必须清空 `qdrant_data/` 重新摄入**（向量维度从 128 → 320 不兼容）
- XGBoost 微调模型初始未训练，仅使用 IDW 融合结果
- Layer 2 接口需要 `OPENAI_API_KEY` 环境变量（建议使用 `.env` 文件管理）
