# Polymarket 数据爬取与分析工具

本项目提供了一套用于从 Polymarket 爬取、分析和可视化市场数据的 Python 工具。它包括自动化数据采集、统计报告生成以及价格走势图表绘制等功能。

## 项目结构

- `crawler.py`: 核心数据爬取引擎。它通过关键词搜索市场，从 Polymarket CLOB（中央限价订单簿）API 获取历史价格数据，并将结果保存至结构化的 CSV 文件。
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

### 2. 分析与可视化 (`judger.py`)

在获取数据集（如 `polymarket_dataset_final.csv`）后，运行此脚本生成报告和图表：

```bash
python judger.py
```

- **统计报告**: 在控制台打印各关键词分布、事件数量及数据密度。
- **可视化**: 生成排名前 5 的活跃市场价格走势图，并保存至 `market_plots/` 目录。

### 3. 交互式数据提取 (`test.py`)

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
