---
name: update-docs
description: "MUST use after ANY of: (1) creating or updating a plan, (2) modifying code, (3) running tests or receiving test results from user, (4) discovering or fixing bugs, (5) any conversation end where files were changed. Proactively invoke — do not wait for user to ask. Maintains record/project/index.md + features/*.md with commit-level change tracking."
---

# update-docs

## 触发场景
- **Plan**：产出实施计划后
- **Execute**：代码修改完成后
- **Test**：运行测试后（包括用户粘贴的测试结果）
- **Bug**：发现或修复 Bug 后
- **对话结束**：任何对话结束时，如有文件变更

不要等用户要求，主动执行。

## 执行步骤

### 1. 识别涉及的功能

- 运行 `git diff --stat` 或 `git status` 查看变更文件
- 读 `record/project/index.md` 功能清单表，根据「核心文件」列匹配变更文件
- 一个文件可能关联多个功能，一个功能可能涉及多个文件
- 匹配不到时：
  - 同目录下的新文件 → 扩展现有功能的核心文件列表
  - 全新模块 → 新建 feature 文件 + 更新 index.md
  - 配置/CI 等辅助文件 → 不需要对应功能

### 2. 更新功能详情文件

读取对应 `record/project/features/fXX-*.md`，更新两个部分：

**「状态」节**（刷新为最新——始终反映当前代码状态）：
- 实现状态：📋计划中 → 🔧进行中 → ✅已完成（或 ❌已废弃）
- 核心文件列表：如有新文件，补充到列表
- 功能描述：如实现方式有变，更新描述
- 测试方法：如有新的测试命令，补充

**「变化」节**（在顶部追加新条目，最新在上）：

```markdown
### [tag] YYYY-MM-DD HH:MM — 变更标题 (`commit_hash`)

<details><summary>详情</summary>

**计划**：这次变更要做什么（概括 + 详情）
**代码修改**：具体修改了哪些文件/函数/逻辑
**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| 测试命令或描述 | ✅/❌ | 说明 |

</details>
```

Tag 选择：
- `[计划]` — 产出计划时
- `[实现]` — 新功能首次实现
- `[修改]` — 对已有功能的改动
- `[修复]` — Bug 修复
- `[重构]` — 重构不改变行为
- `[弃用]` — 标记功能弃用
- `[启用]` — 重新启用弃用的功能

**跨功能测试**：如果一次测试涉及多个功能，在每个涉及的功能文件中都记录相同的测试结果。

**用户粘贴的测试结果**：用户在命令行中运行测试后粘贴结果，同样需要记录到对应功能的变化条目中。

### 3. 增量更新 index.md

只修改涉及的行，不重新生成整个表：
- 更新「最后变更」列的时间
- 更新「状态」列（如状态发生变化）
- 如有新功能，追加新行到功能清单表
- 全局问题汇总表从 features/*.md 的已知问题中聚合（加链接到来源功能文件），不单独编写

### 4. Commit

```bash
git add record/ [其他变更文件]
git commit -m "准确描述本次变更的 commit message"
```

commit message 根据本次变更内容生成，准确描述做了什么。commit 后在变化条目中补充 commit hash。

## 功能详情文件完整模板

新建功能时使用此模板生成 `record/project/features/fXX-kebab-name.md`：

```markdown
# [FXX] 功能名称

## 状态

- **实现状态**：✅已完成 / 🔧进行中 / 📋计划中 / ❌已废弃
- **核心文件**：
  - `path/file.py:行号` — 类名/函数名，职责描述
  - （一个功能可以包含多个文件，全部列出）
- **功能描述**：当前功能做什么，怎么做。
- **测试方法**：可执行的测试命令或代码片段。

## 变化

### [tag] YYYY-MM-DD HH:MM — 变更标题 (`commit_hash`)

<details><summary>详情</summary>

**计划**：这次变更要做什么
**代码修改**：具体修改了什么
**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|

</details>
```

## index.md 功能清单表格式

| ID | 功能 | 核心文件 | 最初实现 | 最后变更 | 状态 | 维护 | 详情 |
|----|------|----------|----------|----------|------|------|------|

核心文件列：列出功能涉及的主要文件/目录（多个用逗号分隔）。

状态图例：
- ✅无误 / ⚠️有注意事项 / 🔴有Bug
- 🟢在用 / 🟡冻结 / 🔴弃用

## 格式约定

- 时间戳：`YYYY-MM-DD HH:MM` (UTC+8 北京时间)
- 功能 ID：FXX 递增（读取 index.md 当前最大 ID 后 +1）
- 文件名：`fXX-kebab-case.md`
- index.md 日期列简写：`MM-DD HH:MM`
- 变化条目按时间倒序排列（最新在上）
- 用 `<details><summary>` 折叠详情内容
