# 小易时序 RAG 预测微服务

基于向量检索（RAG）和逆距离加权（IDW）融合算法的时序预测微服务。运行时无需任何外部服务，数据完全存储在本地磁盘。

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 向量数据库 | Qdrant（本地磁盘模式） |
| 推理引擎 | ONNX Runtime（PyTorch 1D-CNN 导出） |
| 融合排序 | IDW + XGBoost 微调 |
| 数据处理 | NumPy（Z-Score 归一化） |

## 项目结构

```
xiaoyi-query/
├── core/
│   └── processor.py      # TSProcessor - Z-Score 归一化/反归一化
├── components/
│   ├── encoder.py        # ONNXEncoder - PyTorch 1D-CNN 向量化引擎
│   ├── retriever.py      # QdrantRetriever - 本地向量存储与检索
│   └── ranker.py         # FusionRanker - IDW 融合 + XGBoost 微调
├── api/
│   └── schemas.py        # Pydantic 请求/响应数据模型
├── main.py               # FastAPI 入口 + 依赖注入
├── main.py               # FastAPI 入口 + 依赖注入
├── scripts/
│   └── ingest_ett.py    # ETT数据批量摄入工具
└── README.md
```

## 核心模块详解

### 1. TSProcessor（core/processor.py）

实现实例级 Z-Score 归一化，即每个序列独立计算均值 μ 和标准差 σ：

- `normalize(sequence)` — 计算 μ, σ，返回归一化数组及统计量
- `denormalize(sequence, mu, sigma)` — 将预测结果还原为真实量级

### 2. ONNXEncoder（components/encoder.py）

- `__init__` — 使用 PyTorch 动态构建 1D-CNN 模型，随机初始化权重后立即导出为 `encoder_v1.onnx`，随后使用 ONNX Runtime 加载推理
- 模型结构：`Conv1d(1→32, kernel=3) → ReLU → AdaptiveAvgPool1d → Linear(32→128) → L2 Normalize`
- `encode(sequence)` — 输入长度为 100 的归一化数组，输出 128 维嵌入向量

### 3. QdrantRetriever（components/retriever.py）

- 使用 `qdrant_client.QdrantClient(path="./qdrant_data")` 纯本地存储
- `create_collection_if_not_exists()` — 创建使用 Cosine 距离的 collection
- `ingest_batch()` — 批量写入，payload 包含 `future_y`、`mu`、`sigma`
- `search()` — 返回 Top-K 最相似记录及其 Payload 和 score

### 4. FusionRanker（components/ranker.py）

接收检索返回的 Top-K 结果集（包含 score 和对应的 future_y），分两阶段融合：

1. **IDW 融合**（主阶段）— 根据相似度分数计算权重，加权求和得到初步预测
2. **XGBoost 微调**（辅助阶段）— 用 `[distance, mu, sigma]` 构建特征，对 IDW 结果做微调修正

权重计算公式：w_i = 1 / (d_i^2 + ε)

最终预测：y_hat = Σ (w_i / Σ w_j) * future_y_i

## API 接口

### 健康检查

```
GET /health
```

返回服务状态、collection 名称、向量数量和嵌入维度。

### 摄入数据

```
POST /ingest
Content-Type: application/json

{
  "history_x": [1.0, 2.0, ...],   // 长度必须为 100
  "future_y": [6.0, 7.0, ...]    // 任意长度
}
```

流程：`normalize(history_x)` → `encode()` → Qdrant 存储（含 payload）

### 预测

```
POST /predict
Content-Type: application/json

{
  "history_x": [5.0, 6.0, ...],   // 长度必须为 100
  "top_k": 5                       // 检索的最近邻数量（默认 5）
}
```

流程：`normalize(history_x)` → `encode()` → Qdrant search → IDW 融合 → `denormalize()` → 返回预测结果

### 统计信息

```
GET /stats
```

返回向量总数和 collection 详细信息。

## 安装与启动

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

服务启动时会自动：
- 构建 PyTorch 1D-CNN 模型并导出为 `encoder_v1.onnx`
- 创建 Qdrant 本地 collection（存储在 `./qdrant_data` 目录）

### 3. 访问文档

- Swagger UI：http://127.0.0.1:8000/docs
- ReDoc：http://127.0.0.1:8000/redoc

## 数据摄入（ETT 数据集）

### ETT 数据说明

ETT（Electricity Transformer Temperature）是电力变压器温度数据集，每15分钟采样一次，字段包括：

| 字段 | 含义 |
|------|------|
| `date` | 时间戳 |
| `HUFL/HULL/MUFL/MULL/LUFL/LULL` | 各负载级别的高/低油温特征 |
| `OT` | 油温（Oil Temperature，主预测目标） |

### 步骤 1：预览数据（Dry Run）

不写入数据库，仅预览摄入计划：

```bash
# 预览 ETTm1 的 OT 列
python scripts/ingest_ett.py --file ETTm1 --column OT --dry-run

# 预览所有列的摄入计划
python scripts/ingest_ett.py --file ETTm1 --dry-run
```

### 步骤 2：执行摄入

启动服务后（新开一个终端），运行摄入脚本：

```bash
# 摄入 ETTm1 的 OT 列（油温，最常用）
python scripts/ingest_ett.py --file ETTm1 --column OT

# 摄入 ETTm1 全部 7 列（耗时较长，约 5~10 分钟）
python scripts/ingest_ett.py --file ETTm1

# 摄入 ETTm2 的 OT 列
python scripts/ingest_ett.py --file ETTm2 --column OT
```

**摄入参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history-len` | 100 | 历史窗口长度（必须与模型一致） |
| `--future-len` | 48 | 预测窗口长度（4小时×12=48个点） |
| `--batch-size` | 500 | 每批写入Qdrant的条数 |

摄入过程中会显示进度，例如：

```
────────────────────────────────────────────────────────────
摄入列: OT
────────────────────────────────────────────────────────────
  列名: OT
  序列长度: 69653
  滑动窗口: history=100, future=48
  可生成样本数: 69506
    批次 1/139: +500 条 (累计 500/69506)
    批次 2/139: +500 条 (累计 1000/69506)
    ...
```

### 摄入量估算

以 `--history-len 100 --future-len 48` 为例：

| 数据集 | 原始行数 | 可生成样本数 |
|--------|----------|-------------|
| ETTm1 (单列 OT) | 69,653 | ~69,506 |
| ETTm1 (全部7列) | 69,653 | ~486,542 |
| ETTm2 (单列 OT) | 69,653 | ~69,506 |

## 预测与测试

### 1. 检查向量数量

摄入完成后，通过 API 确认数据已写入：

```bash
curl http://127.0.0.1:8000/health
```

响应示例：

```json
{
  "status": "healthy",
  "collection_name": "time_series_rag",
  "vector_count": 69506,
  "embedding_dim": 128
}
```

### 2. 单条预测测试

使用历史数据中某段序列进行预测：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history_x": [30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5,
                  35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5,
                  40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5,
                  45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5,
                  50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5,
                  55.0, 55.5, 56.0, 56.5, 57.0, 57.5, 58.0, 58.5, 59.0, 59.5,
                  60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5, 64.0, 64.5,
                  65.0, 65.5, 66.0, 66.5, 67.0, 67.5, 68.0, 68.5, 69.0, 69.5,
                  70.0, 70.5, 71.0, 71.5, 72.0, 72.5, 73.0, 73.5, 74.0, 74.5,
                  75.0, 75.5, 76.0, 76.5, 77.0, 77.5, 78.0, 78.5, 79.0, 79.5],
    "top_k": 5
  }'
```

### 3. 从 ETT 数据中提取真实样本进行测试

使用 Python 从 ETT CSV 中提取一段真实的历史-未来配对，对比预测值与真实值：

```python
import pandas as pd
import requests

df = pd.read_csv("ETT_data/ETTm1.csv")
column = "OT"
history_len = 100
future_len = 48

# 取数据集中段的真实样本
history_x = df[column].iloc[1000:1100].tolist()
true_future = df[column].iloc[1100:1148].tolist()

response = requests.post("http://127.0.0.1:8000/predict", json={
    "history_x": history_x,
    "top_k": 5
})

prediction = response.json()["prediction"]

# 计算误差
import numpy as np
mae = np.mean(np.abs(np.array(prediction) - np.array(true_future)))
rmse = np.sqrt(np.mean((np.array(prediction) - np.array(true_future)) ** 2))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```

### 4. 不同 top_k 对比

测试不同检索数量对预测效果的影响：

```python
import pandas as pd
import requests
import numpy as np

df = pd.read_csv("ETT_data/ETTm1.csv")
column = "OT"
history_x = df[column].iloc[2000:2100].tolist()
true_future = df[column].iloc[2100:2148].tolist()

for k in [1, 3, 5, 10, 20]:
    resp = requests.post("http://127.0.0.1:8000/predict",
                         json={"history_x": history_x, "top_k": k})
    pred = resp.json()["prediction"]
    mae = np.mean(np.abs(np.array(pred) - np.array(true_future)))
    print(f"top_k={k:2d}  MAE={mae:.4f}")
```

## 使用示例

### 摄入数据

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "history_x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                  21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                  51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                  61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                  71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                  81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                  91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
    "future_y": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
  }'
```

### 预测

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history_x": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                  12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
                  22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
                  42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0,
                  52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0,
                  62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
                  72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0,
                  82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0,
                  92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0],
    "top_k": 5
  }'
```

## 注意事项

- `history_x` 长度必须为 **100**（由 `INPUT_LENGTH` 配置）
- 所有数据存储在本地 `./qdrant_data` 目录，删除该目录将清除所有已摄入数据
- `encoder_v1.onnx` 在首次启动时自动生成，删除后重启服务会重新生成
- XGBoost 微调模型在初始状态未训练（`_is_fitted = False`），仅使用 IDW 融合结果；随着 `/ingest` 数据积累，可调用 `FusionRanker.fit()` 训练微调模型
