# Retrieve 脚本说明

## 概览

本目录包含一个基于 GDELT 数据和 Embedding 排序的新闻检索脚本。

主脚本：
- retrieve_market_news_daily.py

当前范围：
- 仅做新闻检索
- 控制台交互输入
- Embedding 优先排序（不走降级模式）
- 按天输出，便于后续准确度评估

## 脚本做了什么

retrieve_market_news_daily.py 的执行流程：

1. 通过控制台交互读取检索配置
2. 从 DuckDB（gkg_news 表）拉取候选新闻
3. 用 SentenceTransformer 计算语义相似度
4. 将 Embedding 分数与词法分数融合为 hybrid 分数
5. 按最小语义阈值过滤
6. 每天保留 Top K 结果
7. 输出明细、日报汇总和 Markdown 报告

## 环境依赖

推荐 Python 版本：
- Python 3.10 或 3.11

必需包：
- sentence-transformers
- torch
- transformers
- scikit-learn
- pandas
- duckdb

安装示例：

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -U torch torchvision torchaudio
python -m pip install -U sentence-transformers transformers scikit-learn pandas duckdb
```

如果 torch 安装有问题，可尝试 CPU 轮子：

```bash
python -m pip install -U --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## 运行方式

在仓库根目录执行：

```bash
python Retrieve/retrieve_market_news_daily.py
```

脚本会依次提示输入：
- search_query
- start_date（默认：2026-02-25 00:00:00）
- end_date（默认：2026-03-25 23:59:59）
- db_path
- output_dir
- keyword_hints（逗号分隔）
- top_k_day
- max_candidates
- lexical_topn
- embedding_batch_size
- model_name
- hybrid_alpha
- min_embedding_sim
- fallback_keep_ratio

英文环境推荐模型：
- 默认：BAAI/bge-large-en-v1.5（更大模型，通常比 all-MiniLM-L6-v2 更准）
- 可选：intfloat/e5-large-v2

说明：
- 当模型名包含 bge/e5 时，脚本会自动使用对应的输入前缀格式，以提升检索效果。

## 输入字段说明

- search_query：你要检索的新闻主题描述
- keyword_hints：可选，用于先做词法缩小候选集
- hybrid_alpha：Embedding 分数在最终排序中的权重
  - 1.0 表示只看 Embedding 分数
  - 0.0 表示只看词法分数
- min_embedding_sim：语义过滤阈值
- top_k_day：每天保留的结果条数

## 输出文件

脚本会在 output_dir（默认 Retrieve）输出三个文件：

1. embedding_daily_news_<slug>_<date>.csv
- 按天排序后的检索明细
- 含 relevance_label 和 review_notes，方便人工标注

2. embedding_daily_summary_<slug>_<date>.csv
- 每日聚合统计

3. embedding_retrieval_report_<slug>_<date>.md
- 本次运行配置与结果摘要

## 明细 CSV 字段

- date_utc
- rank_in_day
- datetime_utc
- hybrid_score
- embedding_sim
- lexical_sim
- tone
- source
- title
- url
- relevance_label
- review_notes

## 推荐评估流程

1. 在 relevance_label 中人工打标：
- 1 = 相关
- 0 = 不相关

2. 按天计算：
- Precision@5
- Precision@10
- Precision@20

3. 对所有天做宏平均，得到更稳定的检索质量评估。

## 常见错误

### sentence-transformers unavailable

原因：
- 当前环境未安装 sentence-transformers
- Python 版本或 torch 安装不兼容

解决：
- 使用干净的 Python 3.10/3.11 环境
- 重装 torch 与 sentence-transformers

### No candidate articles found

原因：
- 日期窗口内没有数据
- 查询词或 hints 过于严格

解决：
- 扩大时间范围
- 放宽或去掉 keyword_hints
- 提高 max_candidates

## 数据前提

脚本默认 DuckDB 中存在表：
- gkg_news

使用列：
- DATE
- SourceCommonName
- DocumentIdentifier
- Estimated_Title
- V2Tone

## 备注

- 当 Estimated_Title 缺失时，会尝试从 URL slug 恢复标题。
- 按天分组统一使用 UTC。
