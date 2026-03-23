# [F05] CLI 与批量实验

> 最后更新：2026-03-22 17:30 (UTC+8)

## 概述
项目入口层。run.py 通过 argparse 统一管理所有模型和数据集的训练/测试配置，自动推导数据路径和频率。run_all.sh 批量运行 3 模型 × 4 数据集 × 4 预测长度 = 48 组实验。

## 实现
- **状态**：✅已完成
- **核心文件**：
  - `forecast/run.py:6` — main() 函数，argparse 参数定义 + 自动推导 + 实验构建
  - `forecast/scripts/run_all.sh:1` — 批量实验脚本（三重嵌套循环）
- **实现方式**：
  - **参数分组**：基础(model/is_training)、数据(data/root_path/features/target/freq)、预测(seq_len/label_len/pred_len)、模型特定(DLinear: individual; PatchTST: d_model/n_heads/e_layers/patch_len等; Sundial: sundial_model)、优化(epochs/batch_size/patience/lr)、GPU(use_gpu/gpu)、Embedding(embed)
  - **自动推导**：`data_freq_map` 根据数据集名称设置 data_path 和 freq
  - **Sundial 设备**：独立的 CUDA→MPS→CPU 优先级链
  - **实验命名**：`{model}_{data}_{features}_sl{seq_len}_pl{pred_len}`
  - **训练/测试分流**：`if args.model == 'Sundial'` 跳过训练
  - **run_all.sh**：固定 seq_len=96, label_len=48, train_epochs=10；DLinear/PatchTST is_training=1，Sundial is_training=0

## 测试
### 测试方法
```bash
# 验证参数解析
python -m forecast.run --help

# 快速端到端验证
python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1

# 验证脚本语法
bash -n forecast/scripts/run_all.sh && echo "Syntax OK"
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
| 2026-03-22 | 初始实现（CLI 入口 + 批量脚本） | prediction-fusion 分支 |
