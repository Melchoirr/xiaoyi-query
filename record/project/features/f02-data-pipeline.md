# [F02] 数据管道

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/data_provider/data_loader.py:9` — Dataset_ETT_hour（ETTh1/h2）
  - `forecast/data_provider/data_loader.py:94` — Dataset_ETT_minute（ETTm1/m2）
  - `forecast/data_provider/data_factory.py:1` — data_provider() 工厂函数
  - `forecast/utils/timefeatures.py:7` — TimeFeature 体系 + time_features() 入口
- **功能描述**：ETT 数据集完整输入管道。CSV 加载 → 12/4/8 月标准分割 → StandardScaler 归一化（train fit, 全部 transform）→ 时间特征编码（timeenc=0 手动提取 / timeenc=1 频率映射）→ 滑动窗口采样 → DataLoader。支持 M/S/MS 特征模式，返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)。
- **测试方法**：
  ```python
  from forecast.data_provider.data_loader import Dataset_ETT_hour
  ds = Dataset_ETT_hour(root_path='./ETT_data/', data_path='ETTh1.csv',
      flag='train', size=[96, 48, 96], features='M')
  print(f"Train samples: {len(ds)}")  # 期望 8449
  ```

## 变化

### [实现] 2026-03-22 17:30 — 初始实现 (`90d939e`)

<details><summary>详情</summary>

**计划**：实现 ETT 数据集标准加载管道，支持小时和分钟两种频率，对齐学术基准分割。
**代码修改**：
- 新增 `data_provider/data_loader.py`：Dataset_ETT_hour + Dataset_ETT_minute，硬编码 12/4/8 月分割
- 新增 `data_provider/data_factory.py`：data_dict 映射 + data_provider() 工厂
- 新增 `utils/timefeatures.py`：7 个 TimeFeature 子类（归一化到 [-0.5, 0.5]），频率→特征映射

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

</details>
