# Polymarket 数据爬取与分析工具

本项目提供了一套用于从 Polymarket 爬取、分析和可视化市场数据的 Python 工具。它包括自动化数据采集、统计报告生成以及价格走势图表绘制等功能。

## 项目结构

- `crawler.py`: 核心数据爬取引擎。它通过关键词搜索市场，从 Polymarket CLOB（中央限价订单簿）API 获取历史价格数据，并将结果保存至结构化的 CSV 文件。
- `craweler_long.py`: 长周期数据爬虫。针对长期事件进行并发抓取和时间序列对齐，默认只保留在 2026 年内有效历史跨度不少于 90 天的市场，并输出长线数据集。
- `judger.py`: 数据分析与可视化工具。读取爬虫生成的 CSV 文件，提供统计摘要（事件计数、缺失数据分析等），并为最活跃的市场生成时间序列价格图表。
- `test.py`: 命令行交互工具。用于手动搜索特定事件、选择市场/选项，并定义自定义时间范围来精准提取特定 Token 的历史数据。

## 环境要求

确保已安装 Python，然后安装所需的依赖：

```bash
pip install requests pandas matplotlib seaborn tqdm
```

## 使用说明

### 1. 数据采集 (`crawler.py`)

直接在 `crawler.py` 的 `⚙️ 配置区域` 中设置您的搜索关键词和时间范围：

- `KEYWORDS`: 要搜索的术语列表（例如：`["trump", "fed", "bitcoin"]`）。
- `START_DATE` / `END_DATE`: 历史数据的时间范围。
- `GRANULARITY`: 数据分辨率（支持 `1m`, `10m`, `1h`, `1d`）。
- `OUTPUT_CSV`: 采集数据的保存路径。

🛡️ 事件生命周期过滤参数 (双保险：API级 + 本地级)
格式：ISO 8601 (例如 "2025-01-01T00:00:00Z")，不填留空 "" 即可

- EVENT_START_TIME_MIN = ""
- EVENT_START_TIME_MAX = "2025-03-30T00:00:00Z"
- EVENT_CLOSED_TIME_MIN = ""
- EVENT_CLOSED_TIME_MAX = ""

运行脚本：
```bash
python crawler.py
```

### 2. 长周期数据采集 (`craweler_long.py`)

当您需要构建“长期有效市场”数据集时，使用该脚本。其特点：

- 关键词批量搜索后，筛选满足 `TARGET_YEAR` 内有效重叠天数的事件。
- 默认门槛为 `MIN_HISTORICAL_DAYS = 90`（即目标年份内至少约 3 个月可用历史）。
- 采用 `ThreadPoolExecutor` 并发抓取，按 `CHUNK_DAYS` 分片请求 CLOB 历史接口。
- 抓取后会统一重采样到固定网格（`1m` / `10m` / `1h` / `1d`），并做前后向填充，保证时序连续性。

关键配置项（位于 `⚙️ 配置区域`）：

- `KEYWORDS`: 长线主题关键词（默认包含 crypto / AI / 大厂等词）。
- `TARGET_YEAR`: 目标筛选年份（默认 `2026`）。
- `MIN_HISTORICAL_DAYS`: 目标年份内最低有效历史天数（默认 `90`）。
- `GRANULARITY`: 数据粒度（`1m`, `10m`, `1h`, `1d`）。
- `MAX_WORKERS`: 并发线程数（默认 `3`）。
- `OUTPUT_CSV`: 输出文件（默认 `polymarket_real_longterm_dataset.csv`）。
- `LOG_FILE`: 运行日志（默认 `polymarket_spider.log`）。

运行脚本：

```bash
python craweler_long.py
```

说明：当前仓库中的文件名为 `craweler_long.py`（不是 `crawler_long.py`）。

### 3. 分析与可视化 (`judger.py`)

在获取数据集（如 `polymarket_dataset_final.csv`）后，运行此脚本生成报告和图表：

```bash
python judger.py
```

- **统计报告**: 在控制台打印各关键词分布、事件数量及数据密度。
- **可视化**: 生成排名前 5 的活跃市场价格走势图，并保存至 `market_plots/` 目录。

### 4. 交互式数据提取 (`test.py`)

如果您想针对特定 Token 进行测试：

```bash
python test.py
```
按照命令行提示搜索事件、选择市场和选项，并定义时间范围。

## 数据字段说明

保存的 CSV 文件包含以下列：
- `event_title`: 父级事件名称。
- `market_question`: 具体市场问题。
- `outcome`: 投注选项（如 "Yes", "No"）。
- `datetime_utc`: 数据点的 UTC 时间戳。
- `price`: 概率价格（0 到 1 之间）。
- `keyword`: 用于找到该事件的搜索词。
- `token_id`: 选项 Token 的唯一标识符。

## 目录管理

- `market_plots/`: 存储生成的走势图。
- `polymarket_scraper.log`: 记录爬虫运行日志。
