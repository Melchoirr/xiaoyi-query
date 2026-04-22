# GDELT x Polymarket 对齐实验分支

本仓库用于将 GDELT 新闻流与 Polymarket 事件进行对齐，核心流程包括：

1. 事件池构建
2. 新闻召回与排序（关键词 + embedding）
3. 事件级/时间级相关性分析与可视化

---

## 目录总览

- analysis: 分析与实验脚本
- data: 数据下载、CSV/DuckDB 导入与处理
- gdelt_gkg_zips: GDELT 原始压缩数据
- result: 结果产物（按日期归档）
- Retrieve: 检索实验与模型阈值对比（含 round2 报告）

---

## analysis 脚本整理

以下按功能分组整理当前 analysis 下的脚本。

### A. 事件池与面板数据构建

- build_event_pool.py: 构建可与新闻标题匹配的事件池
- build_top10_panel_dataset.py: 构建 top10 事件统一面板数据
- build_model_dataset_from_panel.py: 从对齐面板生成监督学习数据

### B. 基础检索与阈值实验

- query.py: 查询/调试检索逻辑
- run_threshold_study_qwen3.py: 使用 Qwen embedding 进行分层阈值扫描
- plot_qwen_threshold_visuals.py: 将阈值扫描结果画成趋势图、热力图并生成报告

### C. 事件级相关性与批处理可视化

- plot_single_event_qwen_correlation.py: 单事件新闻-价格相关性图与统计输出
- plot_qwen_all_tech_event_correlations.py: 对全部 tech 事件批量跑单事件相关性脚本
- plot_qwen_all_tech_like_small.py: 用 polymarketAlignedGDELTSmall.py 的风格批量画图
- polymarketAlignedGDELTSmall.py: 参考图形风格模板脚本

### D. 对齐统计与波动性分析

- alignment_study_from_panel.py: 从 panel 计算 lead-lag、回归与相关性研究
- visualize_alignment.py: 对齐结果可视化
- lead_lag_analysis.py: lead-lag 关系分析
- vector_volatility.py: 向量相似度与波动率关系分析
- vector_volume_vs_volatility.py: 新闻量与波动率关系分析
- volatility_attention.py: 波动注意力特征分析
- plot_timewise_news_volatility.py: 时间维度新闻冲击与波动图
- validate_news_spike_volatility_hypothesis.py: 验证“新闻峰值先于大波动”假设

---

## result 结果整理

### 1) 主结果目录（日期归档）

当前有效日期目录为 result/by_date/2026-04-21，主要内容：

- 事件池与面板：
	- event_pool_top10_2026-04-21.csv / .md
	- event_pool_top10_2026-04-21_tech.csv / .md
	- event_panel_top10_2026-04-21.csv
	- event_panel_top10_2026-04-21_tech.csv
	- event_panel_summary_top10_2026-04-21.csv / .md
	- event_panel_summary_top10_2026-04-21_tech.csv / .md

- 新闻召回与排序结果：
	- news_retrieval_ranked_top10_2026-04-21.csv
	- news_retrieval_ranked_top10_2026-04-21_tech.csv
	- news_retrieval_ranked_preview_top10_2026-04-21.csv
	- news_retrieval_ranked_preview_top10_2026-04-21_tech.csv
	- news_retrieval_kept_event_ranking_2026-04-21.csv
	- tech_news_retrieval_kept_ranking_2026-04-21.csv

### 2) Qwen 阈值扫描结果

目录：result/by_date/2026-04-21/threshold_study_qwen3_2026-04-21

- 总结文件：
	- threshold_study_summary_2026-04-21.csv / .md
	- threshold_low_range_diagnosis_2026-04-21.csv
- 各阈值子目录（0.2 到 0.7）：
	- 每个阈值目录均包含 event_panel_summary、kept_news_preview、kept_ranking_by_event
- 可视化：
	- visuals/qwen_threshold_total_kept_news.png
	- visuals/qwen_threshold_events_with_news.png
	- visuals/qwen_threshold_event_heatmap.png
	- visuals/qwen_visualization_report_2026-04-21.md

### 3) 全 tech 事件批处理相关性结果

目录：result/by_date/2026-04-21/qwen_all_tech_event_analysis

- qwen_all_tech_event_summary_2026-04-21.csv
- qwen_all_tech_event_report_2026-04-21.md
- qwen_all_tech_event_failures_2026-04-21.csv

当前批处理报告显示：Success events = 5，Failed events = 0。

### 4) 小图风格批量输出（按 polymarketAlignedGDELTSmall 风格）

目录：result/by_date/2026-04-21/qwen_tech_style_plots

- qwen_tech_style_plot_summary_2026-04-21.csv
- 每个事件生成：
	- *_alignment_result.png
	- *_merged_series.csv

当前汇总：5 个 tech 事件中，3 个事件 status=ok，2 个事件 status=no_price_history。

### 5) 历史与归档

- result/RESULTS_INDEX.md: 结果索引
- result/archive/legacy_misc: 历史图表与历史快照
- result/archive/root_duplicates: 根目录历史重复文件备份

---

## 基于现有结果的评估报告：embedding 召回是否合适

### 问题 1：使用 embedding 召回事件的质量是否合适？

结论：合适，但只能算“可用的宽召回层”，不能直接当最终高精度结果。

数据依据（Qwen 阈值实验）：

- 在 threshold=0.6 时，仅 1/5 事件有保留新闻，total_kept_news=1。
	- 说明高阈值精度高但覆盖严重不足。
- 在 threshold=0.4 时，events_with_kept_news=2，total_kept_news=17。
	- 覆盖仍不足，而且开始出现语义噪声。
- 在 threshold=0.32 时，events_with_kept_news=4，total_kept_news=52。
	- 覆盖与可控噪声之间较平衡。
- 在 threshold=0.22 时，events_with_kept_news=5，total_kept_news=300。
	- 全覆盖，但噪声风险显著提升。

质量样例（来自 threshold=0.4 的保留新闻）：

- 相对相关：
	- Claude Overtakes Chatgpt In Apple App Store
	- Claude Dethrones Chatgpt In App Store
- 明显噪声：
	- Ahold Delhaize Transitions To Cage Free Eggs In The Us By 2032
	- Back In The Cccp

解释：embedding 能抓到“主题相似/词义相近”，但在低阈值下会把“语义片段相近但事件无关”的新闻一并带入。

### 问题 2：仅使用 embedding 是否合适？

结论：不合适。

理由 1：仅 embedding 难以同时兼顾覆盖与纯度。

- 提高阈值，覆盖快速塌缩（0.6 仅保留 1 条）。
- 降低阈值，噪声快速上升（0.4 已出现明显无关样本，0.22 达到 300 条）。

理由 2：在独立 round2 对比里，Qwen 的召回规模大，但纯度信号偏弱。

- 同口径对比（query: Will trump visit china）：
	- Qwen@0.6: semantic_kept_n=736, top50_heuristic_precision=0.62
	- BGE@0.6: semantic_kept_n=408, top50_heuristic_precision=0.72
	- MiniLM@0.6: semantic_kept_n=151, top50_heuristic_precision=0.94

这说明 Qwen 在该设置下更偏“宽召回”，但如果不叠加约束，精度不足以直接用于高可信事件判断。

理由 3：事件价格响应并不总与“新闻量”同步，embedding 单独召回难以保证可交易信号质量。

- 样例：event_255933_ethereum-etf-flows-on-march-11_merged_series.csv
	- 首次价格变化发生在 2026-03-09 14:00 UTC。
	- 在此之前，价格平盘 302 小时，但累计 news_volume=625。
	- 3 月 11 日当天价格区间从 0.29 到 0.675（range=0.385），当天 news_volume=64。

这说明“有新闻”不等于“可解释价格变化”，仅 embedding 召回不够，需要事件约束和市场侧过滤。

---

## 推荐做法（基于当前证据）

建议继续使用“三段式”而不是仅 embedding：

1. 关键词/实体预筛（事件实体、地点、时间窗口）
2. embedding 召回（建议阈值先在 0.32-0.4 区间做任务内校准）
3. 规则与统计后筛（来源去重、时间邻近、市场可解释性检验）

对于高精度用途（训练标签、策略信号），必须叠加实体和时间约束；对于探索性分析，可放宽阈值做宽召回但要保留噪声标注。
