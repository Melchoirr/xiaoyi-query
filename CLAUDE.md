# 小易时序预测融合框架 (xiaoyi-query)

## 文档体系
开始工作前阅读相关文档了解当前状态：
- `record/project/index.md` — 功能索引总览（状态、时间、维护情况）
- `record/project/features/` — 各功能详情文件（状态 + 变化历史）
- `record/project/workflow.md` — 开发工作流方法论

### Skill 使用
- `/catch-up` — 初始化文档或追赶落后的文档状态（扫描代码+git→生成/更新所有文档）
- `/update-docs` — 日常文档更新（Plan阶段标记计划，Execute阶段根据diff记录实现）
- `/audit` — 审计未测试区域和潜在问题，输出优先级排序的待办清单

## 行为规则
1. 使用 `python` 而非 `python3`
2. **对话结束时，如有文件变更（代码或文档），执行 `/update-docs` 更新文档后 commit。commit message 根据实际变更准确描述。**
3. **用户粘贴的测试结果同样需要记录到对应功能文档中**
