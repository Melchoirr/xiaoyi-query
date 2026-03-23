# [F02] 数据管道

> 最后更新：2026-03-22 17:30 (UTC+8)

## 概述
ETT 数据集的完整输入管道：CSV 加载 → train/val/test 标准分割 → StandardScaler 归一化 → 时间特征编码 → 滑动窗口采样 → DataLoader 构建。支持 ETTh1/h2（小时）和 ETTm1/m2（分钟）四个数据集。

## 实现
- **状态**：✅已完成
- **核心文件**：
  - `forecast/data_provider/data_loader.py:9` — Dataset_ETT_hour，小时级 ETT 数据集（train 8640 / val 2880 / test 5760 样本，12/4/8 月分割）
  - `forecast/data_provider/data_loader.py:94` — Dataset_ETT_minute，分钟级 ETT 数据集（样本数 ×4）
  - `forecast/data_provider/data_factory.py:1` — data_provider() 工厂函数，data_dict 映射数据集名→类，构建 DataLoader
  - `forecast/utils/timefeatures.py:7` — 7 个 TimeFeature 子类（归一化到 [-0.5, 0.5]），time_features() 按频率自动选特征组合
- **实现方式**：
  - **分割**：硬编码 border1s/border2s，按 ETT 论文标准 12/4/8 月
  - **归一化**：sklearn.StandardScaler，在 train 集上 fit，transform 全部数据
  - **特征模式**：M（多变量全列）、S（单变量 target 列）、MS（多变量输入→单变量输出）
  - **时间编码**：timeenc=0 手动提取 month/day/weekday/hour(/minute)；timeenc=1 通过 timefeatures.py 的频率映射（Hour→4维, Minute→5维）
  - **__getitem__**：返回 (seq_x[seq_len], seq_y[label_len+pred_len], seq_x_mark, seq_y_mark)
  - **DataLoader**：test 不 shuffle，train/val shuffle，统一 drop_last=True
- **设计决策**：ADR-004

## 测试
### 测试方法
```python
from forecast.data_provider.data_loader import Dataset_ETT_hour
ds = Dataset_ETT_hour(root_path='./ETT_data/', data_path='ETTh1.csv',
    flag='train', size=[96, 48, 96], features='M')
print(f"Train samples: {len(ds)}")
seq_x, seq_y, x_mark, y_mark = ds[0]
print(f"seq_x: {seq_x.shape}, seq_y: {seq_y.shape}")
# 验证分割边界
assert len(ds) == 8640 - 96 - 96 + 1  # 8449
```

### 测试结果
| 日期 (UTC+8) | 方法 | 结果 | 备注 |
|--------------|------|------|------|
| （暂无记录） | | | |

## 问题跟踪
### 已知问题
| 问题 | 优先级 | 发现日期 | 状态 |
|------|--------|----------|------|
| （无） | | | |

### 解决记录
| 问题 | 解决方案 | 解决日期 | 验证 |
|------|----------|----------|------|

## 时间线
| 时间 (UTC+8) | 事件 | commit |
|--------------|------|--------|
| 2026-03-22 | 初始实现（数据加载 + 工厂 + 时间编码） | prediction-fusion 分支 |
