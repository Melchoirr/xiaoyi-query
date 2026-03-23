---
name: update-docs
description: "MUST use after ANY of: (1) creating or updating a plan, (2) modifying code, (3) running tests or receiving test results from user, (4) discovering or fixing bugs, (5) any conversation end where files were changed. Proactively invoke — do not wait for user to ask. Maintains record/project/index.md + features/*.md with commit-level change tracking."
---

# update-docs

## 流程

1. **识别涉及的功能** — 读 `record/project/index.md` 功能清单表，按核心文件列匹配变更文件。无匹配时：同目录新文件→扩展现有功能；全新模块→新建 feature 文件；配置文件→跳过。

2. **更新 features/fXX-*.md** — 两个部分：

   **「状态」节**：刷新为最新（实现状态、核心文件、功能描述、测试方法）。

   **「变化」节**：在顶部追加条目（最新在上）：
   ```markdown
   ### [tag] YYYY-MM-DD HH:MM — 标题 (`commit_hash`)
   <details><summary>详情</summary>

   **计划**：做什么
   **代码修改**：改了什么
   **测试**：
   | 方法 | 结果 | 备注 |
   |------|------|------|

   </details>
   ```
   Tag: `[计划]` `[实现]` `[修改]` `[修复]` `[重构]` `[弃用]` `[启用]`

   跨功能测试：在每个涉及的功能文件中都记录。

3. **增量更新 index.md** — 只改涉及的行（最后变更、状态列）。新功能追加行。全局问题汇总从 features/ 聚合（加链接）。

4. **Commit** — `git add` 变更文件 + 文档，commit message 准确描述变更。commit 后补 hash 到变化条目。

## 约定
- 时间：UTC+8，`YYYY-MM-DD HH:MM`
- ID：FXX 递增，文件名 `fXX-kebab-case.md`
- 状态：✅已完成 / 🔧进行中 / 📋计划中 / ❌已废弃
- 折叠：`<details><summary>` 包裹详情
