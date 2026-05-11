---
name: deliver
description: "生成标准化的 GitHub Issue 成果交付 comment。两种类型：(1) 实验结果交付 `/deliver experiment` — 模型位置、结构功能、实验结果及分析、脚本/log/结果位置、附件清单；(2) 工程进展交付 `/deliver engineering` — 代码位置、结构功能、功能截图/视频、测试情况。触发场景：完成一个实验需要汇报结果时，或完成一个工程功能需要交付进展时。"
---

# deliver — 成果交付 Comment 生成

生成标准化的 GitHub Issue comment，用于实验结果或工程进展的成果交付。

## 用法

- `/deliver experiment` — 实验结果交付
- `/deliver engineering` — 工程进展交付
- `/deliver` — 根据当前分支内容自动判断类型

## 执行步骤

### 1. 收集上下文

- `git log --oneline -20` 查看近期提交
- `git diff main --stat` 查看相对主分支的变更
- 读取相关源码，定位核心代码位置（精确到类/方法和大致行号）
- 读取 `record/project/index.md` 和相关 feature 文件了解功能状态
- 检查 `outputs/` 目录下的结果文件、log、图表

### 2. 确认交付类型

若未指定类型，根据变更内容判断：
- 变更涉及 `forecast/models/`、`forecast/baselines/`、`forecast/fusion/`、`scripts/benchmark/` 且有结果产出 → experiment
- 变更涉及工具、脚本、基础设施、CI 等 → engineering
- 两者兼有 → 分别生成两个 comment

### 3. 生成 Comment

按对应模板生成 Markdown 格式的 comment 内容，输出到终端供用户复制，或直接通过 `gh issue comment` 发布。

### 4. 提醒同步

提醒用户将实验结果和可视化同步到 WPS 共享表格的对应 sheet。

---

## 实验结果交付模板

```markdown
## 实验结果交付

### 代码位置

| 文件 | 类/方法 | 行号 | 职责 |
|------|---------|------|------|
| `path/to/file.py` | `ClassName.method_name` | L42-L80 | 简要描述 |

### 模型/方法结构

简要描述模型架构或方法流程（2-4 句话），包括：
- 输入/输出格式
- 核心算法思路
- 关键超参数

### 实验结果

#### 定量结果

| 模型 | MSE | MAE | 备注 |
|------|-----|-----|------|
| **方法A（top1）** | **0.xxx** | **0.xxx** | 最优 |
| **方法B（top2）** | **0.xxx** | **0.xxx** | 次优 |
| 方法C | 0.xxx | 0.xxx | baseline |

#### 结果分析

- 关键发现 1
- 关键发现 2
- 与 baseline 的对比分析

### 产物位置

| 类型 | 路径 |
|------|------|
| 结果 CSV | `outputs/results/xxx.csv` |
| 训练 Log | `outputs/logs/xxx.log` |
| 可视化图表 | `outputs/figures/xxx/` |
| 训练脚本 | `scripts/train/xxx.sh` |

### 附件

- [ ] 结果 CSV（top1/top2 已标注）
- [ ] 训练 Log
- [ ] 可视化分析图表
- [ ] WPS 共享表格已同步（sheet: ___）
```

## 工程进展交付模板

```markdown
## 工程进展交付

### 代码位置

| 文件 | 类/方法 | 行号 | 职责 |
|------|---------|------|------|
| `path/to/file.py` | `ClassName.method_name` | L42-L80 | 简要描述 |

### 功能结构

简要描述新增/修改功能的结构（2-4 句话），包括：
- 功能入口和调用链
- 核心逻辑
- 与现有代码的集成方式

### 功能展示

（截图或视频，描述新功能的使用效果）

### 测试情况

| 测试项 | 方法 | 结果 | 备注 |
|--------|------|------|------|
| 功能测试 | 描述 | ✅/❌ | |
| 边界情况 | 描述 | ✅/❌ | |
| 兼容性 | 描述 | ✅/❌ | |

### 附件

- [ ] 功能截图/视频
- [ ] 相关测试输出
```

## 注意事项

- 代码位置必须精确到方法和大致行号，不能只写文件名
- 实验结果 CSV 中 top1 和 top2 需要特别标注（加粗或颜色）
- 实验结果和可视化需同步到 WPS 共享表格，按所属模块/实验放到对应 sheet
- Comment 发布后提醒用户检查附件是否完整
