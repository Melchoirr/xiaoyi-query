# Polymarket × GDELT 知识图谱新闻对齐系统

基于 Wikidata 知识图谱，将 Polymarket 预测市场价格与 GDELT 新闻对齐，构建「价格 × 新闻」数据集，为训练 LLM 预测价格变化提供数据支撑。

## 系统架构

```
Polymarket API ──→ 实体抽取 (spaCy) ──→ Wikidata 链接 ──→ 知识图谱
     │                                                        │
     └── 价格曲线                                              ├── Wikidata 关系（行业、产品、地理...）
                                                               │
GDELT GKG (DuckDB) ←── 实体搜索 (P2) ←── 语义过滤 (P0) ←── 跨事件去重 (P1)
     │
     └── 新闻量时序 ──→ 价格-新闻对齐数据集
```

## Pipeline 三步改进

### V1：实体关键词匹配（基线）

基于 spaCy NER + Wikidata 实体链接，从事件标题提取核心实体，用 ILIKE 在 GDELT 新闻标题中匹配。

**问题**：
- 「Iran」匹配所有伊朗新闻，无论是否与具体事件相关
- 「US」作为子串匹配到 "focus"、"virus" 等无关词
- 5 个伊朗事件共享同一批 ~17 万篇文章，缺乏区分度

### V2：P0 + P1 + P2（增强版）

| 改进 | 内容 | 技术 |
|------|------|------|
| **P2** | KG 细粒度实体扩展 | Wikidata 1-2 跳关系实体作为搜索词（如 "Federal Open Market Committee" 替代仅 "Fed"） |
| **P0** | Embedding 语义过滤 | all-MiniLM-L6-v2 计算事件标题与新闻标题的 cosine similarity，阈值 0.35 |
| **P1** | 跨事件 URL 去重 | 同一 URL 只归属到 similarity 最高的事件，消除事件间信号串扰 |

### 噪声削减效果

| 事件 | V1 新闻数 | V2 新闻数 | 降噪率 |
|------|----------|----------|--------|
| Trump 收购格陵兰 | 109,748 | 11,997 | 89.1% |
| 伊朗/以色列/美国冲突 | 185,116 | 74,776 | 59.6% |
| 伊朗打击海湾石油设施 | 185,431 | 51,493 | 72.2% |
| 美伊核协议 | 174,959 | 23,205 | 86.7% |
| 中国入侵台湾 | 18,054 | 7,243 | 59.9% |
| 俄乌停火 | 17,127 | 8,623 | 49.7% |
| Trump 提名美联储主席 | 105,827 | 5,557 | 94.7% |
| 美联储 3 月决议 | 59,269 | 3,532 | 94.0% |
| 美国入侵伊朗 | 171,643 | 19,370 | 88.7% |
| 美联储 4 月决议 | 17,684 | 1,572 | 91.1% |

**P1（去重）贡献最大**：伊朗核协议从 11.6 万 → 2.3 万，其中 9.3 万篇被去重（URL 更匹配其他伊朗事件）。

---

## 数据集评估方法

> 完整分析见 [REPORT.md](REPORT.md) —— 包含详细的特征工程、AR(1) 基线推导、显著性判定、Surge 效应分析及数据集质量评估。

### 核心问题

能否用 t 时刻的已知信息，预测 t+h 时刻的价格变化？

### 特征构建

#### 1. 价格特征

| 特征 | 公式 | 说明 |
|------|------|------|
| `dprice_1h` | price(t) − price(t−1h) | 最近 1 小时价格变化 |
| `dprice_3h` | price(t) − price(t−3h) | 最近 3 小时价格变化 |
| `dprice_6h` | price(t) − price(t−6h) | 最近 6 小时价格变化 |
| `dprice_12h` | price(t) − price(t−12h) | 最近 12 小时价格变化 |
| `dprice_24h` | price(t) − price(t−24h) | 最近 24 小时价格变化 |
| `price_ma_6h` | rolling mean of price over ±3h | 6 小时价格均线 |
| `price_std_6h` | rolling std of price over ±3h | 6 小时价格波动率 |

所有价格字段先做 linear interpolation 再 forward-fill / backward-fill，确保时间序列连续、无缺失。

#### 2. 新闻特征

| 特征 | 公式 | 说明 |
|------|------|------|
| `news_volume` | count(GKG articles) per hour | 该小时匹配的 GDELT 新闻数量 |
| `news_ma_24h` | rolling mean of news_volume over 24h | 24 小时新闻量均值（基准线） |
| `news_zscore_24h` | (news_volume − news_ma_24h) / std_24h | 标准化后的新闻异常度 |
| `news_surge` | news_volume > 2 × news_ma_24h | 布尔标记：新闻突然爆发 |

`news_zscore_24h` 是核心特征。它衡量当前小时的新闻量相对过去 24 小时均值偏离了多少个标准差：

```
z(t) = (V(t) − μ₂₄(t)) / σ₂₄(t)

其中  μ₂₄(t) = (1/24) · Σ V(t−i),  i=1..24
      σ₂₄(t) = rolling std of last 24 hours (min floor = 1.0)
```

标准差取 max(std, 1.0) 避免除零，均值仅 0-1 篇/小时的新闻洼地被抑制。

### AR(1) 基线模型（价格自回归）

#### 为什么用 AR(1)

Polymarket 是一个信息高效市场，价格本身已经聚合了所有公开信息（包括新闻）。AR(1) 回答的问题是：**仅靠价格动量（price momentum），对未来价格变化的预测力有多大？** 新闻信号只有在**超越价格动量**时才有增量价值。

#### 公式

AR(1) 模型假设下一时刻价格变化与当前变化线性相关：

```
Δprice(t+h) = β · dprice_1h(t) + ε

其中  dprice_1h(t) = price(t) − price(t−1h)    ← 当前 1 小时动量
      Δprice(t+h) = price(t+h) − price(t)      ← 未来 h 小时的真实变化
```

评估时不做回归拟合，直接计算 `dprice_1h(t)` 与 `Δprice(t+h)` 的 Pearson 相关系数：

```
AR(1)_r(h) = Pearson(dprice_1h(t), Δprice(t+h))
```

- `dprice_1h(t)` 从当前行取值（已知信息）
- `Δprice(t+h)` 从未来行取值（预测目标），通过 `shift(-h)` 对齐
- 正值表示动量效应（涨→继续涨），负值表示均值回复（涨→会回落）

#### 举个具体例子

假设 t = 2026-03-15 12:00，horizon = 3h：

```
dprice_1h(t)     = price(12:00) − price(11:00) = 0.52 − 0.50 = +0.02
Δprice(t+3h)     = price(15:00) − price(12:00) = 0.56 − 0.52 = +0.04
```

AR(1) 用 +0.02（当前动量）去拟合 +0.04（未来 3 小时变化）。对所有时间点计算这对值的 Pearson r，即为该 horizon 的 AR(1) 预测力。

### 新闻预测力评估

#### 对比方法

对每个 horizon h ∈ {1, 3, 6, 12, 24}，计算三个指标的 Pearson r 与目标 `Δprice(t+h)`：

| 指标 | 公式 | 物理含义 |
|------|------|----------|
| `AR(1)_r` | r(dprice_1h(t), Δprice(t+h)) | 价格动量的预测力 |
| `NewsZ_r` | r(news_zscore_24h(t), Δprice(t+h)) | 新闻异常的预测力 |
| `Vol_r` | r(news_volume(t), Δprice(t+h)) | 新闻绝对量的预测力 |

#### 显著性判定

- p < 0.05 → 标记为 `*`（统计显著）
- `NewsZ_r` > `AR(1)_r` → 新闻超越了价格自回归（有增量信息）
- `NewsZ_r` ≤ `AR(1)_r` → 新闻未提供额外预测力

#### News Surge 效应

计算 news_surge=1 时刻与 news_surge=0 时刻的平均 Δprice 差异：

```
surge_effect = mean(Δprice | surge=1) − mean(Δprice | surge=0)
```

正值表示新闻爆发时价格上涨，负值表示下跌。

### 数据集质量评估

对每个事件的 merged_series CSV，统计：

| 指标 | 含义 |
|------|------|
| `hours` | 总小时数 |
| `mean_vol` | 平均每小时新闻数 |
| `median_vol` | 中位数每小时新闻数 |
| `max_vol` | 单小时最高新闻数 |
| `zero_pct` | 零新闻小时占比（越低越好） |
| `vol_std` | 新闻量标准差（波动性） |

零新闻小时占比是关键指标：如果 50% 以上的小时没有匹配到任何新闻，说明该事件新闻覆盖不足，难以训练预测模型。

### 预测能力分析结果

**伊朗地缘政治事件预测力最强：**

| 事件 | 最佳 horizon | NewsZ_r | AR(1)_r | 解读 |
|------|-------------|---------|---------|------|
| 伊朗/以色列冲突 | 12h | **+0.283*** | −0.075 | 新闻显著超越价格动量 |
| 伊朗打击海湾石油 | 24h | **+0.253*** | −0.004 | 新闻独自提供预测信号 |
| 美伊核协议 | 3h | **+0.168*** | −0.048 | 新闻短期预测力最强 |

**V1→V2 增强后预测力提升：**

| 事件 | Horizon | V1 NewsZ_r | V2 NewsZ_r | 提升 |
|------|---------|-----------|-----------|------|
| 伊朗冲突 | 12h | +0.255 | **+0.283** | +11% |
| 美联储 3 月 | 12h | +0.073 | **+0.114*** | +56% |
| Trump 美联储主席 | 3h | +0.033 | **+0.092*** | +179% |

**低覆盖事件预测力不足：**

| 事件 | 均值新闻/h | 零新闻比例 | 结论 |
|------|-----------|-----------|------|
| 美联储 4 月 | 2.3 | 38% | 无显著预测力 |
| 中国入侵台湾 | 10.4 | — | 信号被 AR 主导 |

### 核心结论

- 新闻 z-score 在 **15/50 个 horizon-事件组合中超越 AR 基线**（V1 为 14/50）
- **伊朗地缘政治事件是当前数据集最强预测场景**，12-24h 新闻信号显著
- AR(1) 为正向动量（正 Pearson r），但伊朗类事件 AR 不显著（价格接近随机游走），新闻正好填补空白
- **美联储和低覆盖事件** AR 自回归占主导，新闻未能提供增量信息

---

## 代码文件

| 文件 | 用途 |
|------|------|
| `polymarket_client.py` | Polymarket API 事件 + 价格曲线拉取 |
| `entity_linker.py` | spaCy NER + Wikidata 实体链接 + SPARQL 关系扩展 |
| `knowledge_graph.py` | 知识图谱构建 + SQLite 存储 |
| `llm_enricher.py` | DeepSeek LLM 补充商业关系（可选） |
| `gdelt_matcher.py` | **V1** 关键词匹配器（DuckDB ILIKE） |
| `gdelt_matcher_v2.py` | **V2** 增强匹配器（P0+P1+P2） |
| `generate_alignment.py` | V1 对齐序列生成 + 双轴图绘制（含 tone） |
| `plot_alignment_v2.py` | V2 对齐图绘制（纯 price + news_volume，无 tone） |
| `analyze_correlation.py` | 定量分析（AR(1) 基线、相关性、预测力评估） |
| `expand_events.py` | 大规模事件扩展 pipeline（50+ 事件） |
| `run_kg_pipeline.py` | 知识图谱构建脚本（离线模式） |
| `main.py` | 完整 pipeline 入口 |

## 输出文件

```
result/by_date/2026-05-11/kg_gdelt_alignment_v2/
├── event_118172_..._merged_series.csv   ← 价格 + 新闻量时序
├── event_118172_..._alignment_v2.png    ← 双轴可视化（无 tone）
├── ...（10 个事件）
└── alignment_summary.csv
```

每个 CSV 包含列：`datetime_utc, price, news_volume`

## 后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| P3 | 增加事件数量（50+），丰富事件类型 | 跨事件泛化，LLM 训练样本多样性 |
| P4 | 价格冲击标注（新闻发布后 Δprice 超过阈值） | LLM 训练正例标签 |
| P5 | 事件标题 + description 作为 embedding query | 更好的语义匹配 |
| P6 | 反事实过滤（Granger causality 判断新闻驱动因素） | 提升单篇新闻的信噪比 |

## Setup

```bash
conda create -n polymarket-alignment python=3.10
conda activate polymarket-alignment
pip install spacy duckdb sentence-transformers pandas numpy scipy matplotlib requests
python -m spacy download en_core_web_sm
```

## Usage

```bash
# V2 增强对齐（P0+P1+P2）
python gdelt_matcher_v2.py 0.35 result/by_date/2026-05-11/kg_gdelt_alignment_v2

# 定量预测力分析
python analyze_correlation.py result/by_date/2026-05-11/kg_gdelt_alignment_v2

# V2 对齐图（无 tone）
python plot_alignment_v2.py result/by_date/2026-05-11/kg_gdelt_alignment_v2

# 大规模事件扩展
python expand_events.py --max-events 30 --output-dir result/by_date/2026-05-11/kg_gdelt_alignment_v3
```
