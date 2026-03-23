# [F05] CLI 与批量实验

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/run.py:6` — main() 函数，argparse 参数定义 + 自动推导 + 实验构建
  - `forecast/scripts/run_all.sh:1` — 批量实验脚本（3模型 × 4数据集 × 4预测长度 = 48 组）
- **功能描述**：项目入口层。run.py 通过 argparse 管理所有配置（基础/数据/预测/模型特定/优化/GPU/Embedding），自动推导 data_path 和 freq，Sundial 跳过训练。run_all.sh 批量运行全部实验组合。
- **测试方法**：
  ```bash
  python -m forecast.run --help
  python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1
  bash -n forecast/scripts/run_all.sh && echo "Syntax OK"
  ```

## 变化

### [实现] 2026-03-22 17:30 — 初始实现 (`90d939e`)

<details><summary>详情</summary>

**计划**：实现统一 CLI 入口和批量实验脚本。
**代码修改**：
- 新增 `run.py`：argparse 参数分组（基础/数据/预测/DLinear/PatchTST/Sundial/优化/GPU/Embedding），data_freq_map 自动推导，Sundial 设备配置，实验命名 setting
- 新增 `scripts/run_all.sh`：三重循环（模型→数据集→pred_len），DLinear/PatchTST is_training=1，Sundial is_training=0

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

</details>
