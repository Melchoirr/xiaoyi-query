# Polymarket × GDELT 知识图谱新闻对齐系统

## 项目概述

构建基于 Wikidata 知识图谱的 GDELT 新闻对齐 pipeline，为 Polymarket 预测市场事件构建「价格 × 新闻」对齐数据集，支撑基于新闻的价格预测模型。

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

## 数据规模

| 维度 | 数值 |
|------|------|
| GDELT 新闻总量 | 3,284,556 篇 (2026-02-25 ~ 2026-03-25) |
| GKG 原始压缩包 | 2,808 个 |
| Polymarket 事件 | 10 个（地缘政治类） |
| 知识图谱节点 | 121 个 (Wikidata Q-ID) |
| 知识图谱边 | 308 条 |

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

## 预测能力分析

### 核心问题

能否用 t 时刻的已知信息，预测 t+h 时刻的价格变化？

评估逻辑：若新闻信号能超越仅靠价格动量的 AR(1) 基线，则新闻提供了增量预测价值——这正是构建「价格 × 新闻」对齐数据集的目标。

### 评估方法

#### 1. 特征构建

##### 价格特征

从 Polymarket 价格曲线（每小时采样），构建以下特征：

| 特征 | 公式 | 说明 |
|------|------|------|
| `dprice_1h` | price(t) − price(t−1h) | 最近 1 小时价格变化（瞬时动量） |
| `dprice_3h` | price(t) − price(t−3h) | 最近 3 小时价格变化 |
| `dprice_6h` | price(t) − price(t−6h) | 最近 6 小时价格变化 |
| `dprice_12h` | price(t) − price(t−12h) | 最近 12 小时价格变化 |
| `dprice_24h` | price(t) − price(t−24h) | 最近 24 小时价格变化 |
| `price_ma_6h` | rolling mean of price over ±3h (window=6) | 6 小时价格均线 |
| `price_ma_24h` | rolling mean of price over ±12h (window=24) | 24 小时价格均线 |
| `price_std_6h` | rolling std of price (window=6) | 6 小时价格波动率 |
| `price_std_24h` | rolling std of price (window=24) | 24 小时价格波动率 |

所有价格字段先做 linear interpolation 再 forward-fill / backward-fill，确保时间序列连续、无缺失。

##### 新闻特征

从 GDELT GKG 新闻匹配结果（每小时聚合），构建以下特征：

| 特征 | 公式 | 说明 |
|------|------|------|
| `news_volume` | count(GKG articles) per hour | 该小时匹配的 GDELT 新闻数量 |
| `news_ma_6h` | rolling mean of news_volume over 6h | 6 小时新闻量均值（短期基准线） |
| `news_ma_24h` | rolling mean of news_volume over 24h | 24 小时新闻量均值（长期基准线） |
| `news_delta_1h` | news_volume(t) − news_volume(t−1h) | 1 小时新闻量变化 |
| `news_delta_6h` | news_volume(t) − news_volume(t−6h) | 6 小时新闻量变化 |
| `news_zscore_24h` | (news_volume − news_ma_24h) / max(std_24h, 1.0) | **核心特征**：标准化后的新闻异常度 |
| `news_surge` | news_volume > 2 × news_ma_24h | 布尔标记：新闻突然爆发 |

`news_zscore_24h` 是核心预测特征。它衡量当前小时的新闻量相对过去 24 小时均值偏离了多少个标准差：

```
z(t) = (V(t) − μ₂₄(t)) / max(σ₂₄(t), 1.0)

其中  μ₂₄(t) = (1/24) · Σ V(t−i),  i=1..24
      σ₂₄(t) = rolling std of last 24 hours
```

标准差取 max(std, 1.0) 避免除零（低新闻量时段 std 接近 0），均值仅 0-1 篇/小时的新闻洼地被自然抑制——这些时段的 z-score 不会异常放大。

#### 2. AR(1) 基线模型（价格自回归）

##### 为什么用 AR(1)

Polymarket 是一个信息高效市场，价格本身已经聚合了所有公开信息（包括新闻）。AR(1) 回答的问题是：**仅靠价格动量（price momentum），对未来价格变化的预测力有多大？**

新闻信号只有在**超越价格动量**时才有增量价值——如果新闻 z-score 的预测力不高于 dprice_1h 的预测力，说明新闻已被市场定价消化，未提供额外信息。

##### 公式推导

AR(1) 模型假设下一时刻价格变化与当前变化线性相关：

```
Δprice(t+h) = β · dprice_1h(t) + ε

其中  dprice_1h(t) = price(t) − price(t−1h)    ← 当前 1 小时动量（已知信息）
      Δprice(t+h) = price(t+h) − price(t)      ← 未来 h 小时的真实变化（预测目标）
```

评估时不拟合回归系数，直接计算 `dprice_1h(t)` 与 `Δprice(t+h)` 的 Pearson 相关系数：

```
AR(1)_r(h) = Pearson(dprice_1h(t), Δprice(t+h))
```

- `dprice_1h(t)` 从当前行取值（t 时刻已知）
- `Δprice(t+h)` 从未来行取值（通过 `shift(-h)` 对齐到同一时间索引）
- r > 0 表示动量效应（涨→继续涨），r < 0 表示均值回复（涨→会回落）

##### 具体计算示例

以 t = 2026-03-15 12:00, horizon = 3h 为例：

```
dprice_1h(t)     = price(12:00) − price(11:00) = 0.52 − 0.50 = +0.02
Δprice(t+3h)     = price(15:00) − price(12:00) = 0.56 − 0.52 = +0.04
```

AR(1) 用 +0.02（当前动量）去拟合 +0.04（未来 3 小时变化）。对所有时间点 t 计算这对值的 Pearson r，即为 horizon=3h 的 AR(1) 预测力。若 r ≈ 0，说明近期价格动量对未来变化无预测能力（价格接近随机游走）。

#### 3. 新闻预测力评估

##### 对比框架

对每个 horizon h ∈ {1, 3, 6, 12, 24}，计算三个指标的 Pearson r 与目标 `Δprice(t+h)`：

| 指标 | 公式 | 物理含义 |
|------|------|----------|
| `AR(1)_r` | r(dprice_1h(t), Δprice(t+h)) | 价格动量对未来变化的预测力 |
| `NewsZ_r` | r(news_zscore_24h(t), Δprice(t+h)) | 新闻异常对未来变化的预测力 |
| `Vol_r` | r(news_volume(t), Δprice(t+h)) | 新闻绝对量对未来变化的预测力 |

`Vol_r` 作为辅助对照：如果 NewsZ_r > Vol_r，说明标准化（相对 24h 基准线的偏离）确实比绝对新闻量更有信息量。

##### 显著性判定

- p < 0.05 → 标记为 `*`（统计显著，拒绝 r=0 的原假设）
- `NewsZ_r` > `AR(1)_r` → 新闻超越了价格自回归，提供了增量预测信息
- `NewsZ_r` ≤ `AR(1)_r` → 新闻未提供额外预测力，价格动量已包含同等或更强的信息

##### News Surge 效应

计算 news_surge=1 时刻与 news_surge=0 时刻的平均 Δprice 差异：

```
surge_effect(h) = mean(Δprice(t+h) | surge(t)=1) − mean(Δprice(t+h) | surge(t)=0)
```

正值表示新闻爆发时未来价格倾向于上涨，负值表示倾向于下跌。这是对「新闻爆发→价格变化」方向的定性判断，不作为严格的预测指标。

#### 4. 数据集质量评估

对每个事件的 merged_series CSV，统计以下指标：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| `hours` | 总小时数 | len(df) |
| `price_ok` | 价格有效小时数 | count(price not null) |
| `mean_vol` | 平均每小时新闻数 | mean(news_volume) |
| `median_vol` | 中位数每小时新闻数 | median(news_volume) |
| `max_vol` | 单小时最高新闻数 | max(news_volume) |
| `zero_pct` | 零新闻小时占比 | count(news_volume==0) / n × 100 |
| `vol_std` | 新闻量标准差 | std(news_volume) |

零新闻小时占比（`zero_pct`）是最关键的覆盖度指标：如果 50% 以上的小时没有任何匹配新闻，该事件的新闻信号本质上过于稀疏，难以训练有效的预测模型。

##### 当前数据集整体质量

| 指标 | 数值 |
|------|------|
| 总覆盖小时数 | ~7,000h（10 个事件合计） |
| 总匹配新闻数 | ~22.7 万篇 |
| 零新闻小时占比 | 取决于事件（从 5% 到 38%） |
| 平均新闻量 | ~32 篇/小时 |

### 关键发现

**1. 伊朗事件新闻预测力最强**

| 事件 | 最佳 horizon | NewsZ_r | AR(1)_r | 新闻优势 |
|------|-------------|---------|---------|---------|
| 伊朗/以色列冲突 | 12h | **+0.283*** | -0.075 | ✅ 显著超越 |
| 伊朗打击海湾石油 | 24h | **+0.253*** (Vol) | -0.004 | ✅ |
| 美伊核协议 | 3h | **+0.168*** (Vol) | -0.048 | ✅ |

伊朗类事件 AR(1) 均不显著（r ≈ 0，p > 0.05），说明 Polymarket 上伊朗相关预测的价格接近随机游走——近期价格动量对远期变化无预测能力。新闻信号正好填补了这一空白，在 12-24 小时 horizon 上提供了显著的增量预测力。

**2. 增强后新闻预测力提升**

| 事件 | Horizon | V1 NewsZ_r | V2 NewsZ_r | 提升 |
|------|---------|-----------|-----------|------|
| 伊朗冲突 | 12h | +0.255 | **+0.283** | +11% |
| 美联储 3 月 | 12h | +0.073 | **+0.114*** | +56% |
| Trump 美联储主席 | 3h | +0.033 | **+0.092*** | +179% |
| 美伊核协议 | 3h | +0.123 | **+0.127** (Vol 更强) | — |

V2 增强（P0 语义过滤 + P1 跨事件去重 + P2 细粒度实体扩展）在所有事件上均提升了 NewsZ_r。提升幅度最大的集中在美联储相关事件，因为这些事件受益于 P2 细粒度实体扩展（如 "Federal Open Market Committee" 替代仅 "Fed"）——减少了噪音匹配，增强了信号纯度。

**3. 低覆盖事件预测力不足**

| 事件 | 均值新闻/h | 零新闻比例 | 最佳预测 | 原因 |
|------|-----------|-----------|---------|------|
| 美联储 4 月 | 2.3 | 38% | 无显著 | 新闻覆盖太低，信号过于稀疏 |
| 中国入侵台湾 | 10.4 | ~15% | 无显著 | 信号被 AR 主导，新闻未提供增量 |
| 俄乌停火 | 12.4 | ~12% | 新闻为反指 | 方向性错误（新闻多时价格反而下跌） |

低覆盖事件的核心瓶颈不是模型方法，而是新闻匹配数量不足——当大部分小时新闻量为 0 时，news_zscore_24h 退化为常数，无法提供有效变异。

### 核心结论

- **新闻 z-score 在 15/50 个 horizon-事件组合中超越 AR 基线**（V1 为 14/50，V2 提升 1 个组合）
- **伊朗地缘政治事件是当前数据集最强预测场景**：AR(1) 不显著（价格随机游走）+ 新闻覆盖充足（50-75 篇/小时）+ 新闻信号显著（r > 0.25, p < 0.05）
- **美联储事件受益于 V2 增强最大**：细粒度实体扩展显著提升了信号纯度，但绝对新闻量仍然偏低
- **低覆盖事件**（<5 篇/小时）需要优先解决新闻匹配问题（P4/P5），而非改进预测模型
- **Surge 效应**在多数事件上为正（新闻爆发时价格短期上涨），但效应量小且不显著，适合作为辅助特征而非核心信号

## 实体链接质量

### 正确的链接（已验证）

| 实体 | Wikidata Q-ID | 验证 |
|------|--------------|------|
| Donald Trump | Q22686 | ✅ |
| Federal Reserve System | Q53536 | ✅ |
| Iran | Q794 | ✅ |
| Israel | Q801 | ✅ |
| China | Q148 | ✅ |
| Taiwan | Q865 | ✅ |
| Russia | Q159 | ✅ |
| Ukraine | Q212 | ✅ |
| United States | Q30 | ✅ |
| Greenland | Q223 | ✅ |

### 已知的局限性

- «Fed» 需通过 SEARCH_TERM_OVERRIDES 映射为 «Federal Reserve System» 才能正确搜索
- «US»/«U.S.» 在标题中被 ILIKE 匹配，但 embedding 过滤有助于排除不相关文章
- Wikidata 关系扩展 (SPARQL) 偶尔引入不相关实体（如从 Iran 扩展到 BRICS、Arctic Council）

## 输出文件

```
result/by_date/2026-05-11/kg_gdelt_alignment_v2/
├── event_118172_will-trump-acquire-greenland-before-2027_merged_series.csv
├── event_236884_iran-x-israelus-conflict-ends-by_merged_series.csv
├── event_237306_will-iran-strike-gulf-oil-facilities-by-march-31_merged_series.csv
├── event_257313_us-iran-nuclear-deal-by-april-30_merged_series.csv
├── event_34044_will-china-invade-taiwan-by-end-of-2026_merged_series.csv
├── event_34050_russia-x-ukraine-ceasefire-by-end-of-2026_merged_series.csv
├── event_35908_who-will-trump-nominate-as-fed-chair_merged_series.csv
├── event_67284_fed-decision-in-march_merged_series.csv
├── event_73130_will-the-us-invade-iran-before-2027_merged_series.csv
├── event_75478_fed-decision-in-april_merged_series.csv
└── alignment_summary.csv
```

每个 CSV 包含列：`datetime_utc, price, news_volume`

## 代码文件

| 文件 | 用途 |
|------|------|
| `polymarket_client.py` | Polymarket API 事件 + 价格曲线拉取 |
| `entity_linker.py` | spaCy NER + Wikidata 实体链接 + SPARQL 关系扩展 |
| `knowledge_graph.py` | 知识图谱构建 + SQLite 存储 |
| `llm_enricher.py` | DeepSeek LLM 补充商业关系（可选） |
| `gdelt_matcher.py` | V1 关键词匹配器（DuckDB ILIKE） |
| `gdelt_matcher_v2.py` | **V2 增强匹配器（P0+P1+P2）** |
| `generate_alignment.py` | 对齐序列生成 + 双轴图绘制 |
| `analyze_correlation.py` | 定量分析（相关性、预测力评估） |
| `run_kg_pipeline.py` | 知识图谱构建脚本 |
| `main.py` | 完整 pipeline 入口 |

## 后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| P3 | 增加事件数量（>50 个），丰富事件类型 | 跨事件泛化 |
| P4 | 事件标题 + description 作为 embedding query（而非仅 title） | 更好的语义匹配 |
| P5 | 训练 LightGBM/XGBoost 预测模型（价格 + 新闻特征） | 实际预测应用 |
| P6 | 新闻情感分析（虽然当前未使用，但可补充信号） | 辅助特征 |
