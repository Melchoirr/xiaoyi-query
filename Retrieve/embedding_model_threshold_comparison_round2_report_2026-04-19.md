# Embedding 第二轮严格对比报告

## 实验设置

- query: Will trump visit china
- window_utc: 2026-02-25 00:00:00 to 2026-03-25 23:59:59
- models: all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5
- supplemental validation model: Qwen/Qwen3-Embedding-0.6B (threshold=0.6)
- thresholds: [0.45, 0.5, 0.55, 0.6]
- stage1 max_candidates: 120000
- stage2 lexical_topn: 5000
- stage3 hybrid_alpha: 0.75
- stage3 embedding_batch_size: 128

## 对比结果

| model_name             |   min_embedding_sim |   candidate_n |   lexical_prefilter_n |   semantic_kept_n |   top50_unique_titles |   top50_unique_sources |   top50_dup_ratio |   top50_mean_hybrid |   top50_mean_embedding |   top50_heuristic_precision |
|:-----------------------|--------------------:|--------------:|----------------------:|------------------:|----------------------:|-----------------------:|------------------:|--------------------:|-----------------------:|----------------------------:|
| BAAI/bge-large-en-v1.5 |                0.45 |        119855 |                  5000 |              3162 |                    42 |                     39 |              0.16 |            0.592717 |               0.690596 |                        0.7  |
| BAAI/bge-large-en-v1.5 |                0.5  |        119855 |                  5000 |              2090 |                    42 |                     39 |              0.16 |            0.592717 |               0.690596 |                        0.7  |
| BAAI/bge-large-en-v1.5 |                0.55 |        119855 |                  5000 |               963 |                    42 |                     39 |              0.16 |            0.592717 |               0.690596 |                        0.7  |
| BAAI/bge-large-en-v1.5 |                0.6  |        119855 |                  5000 |               408 |                    41 |                     39 |              0.18 |            0.592492 |               0.693269 |                        0.72 |
| all-MiniLM-L6-v2       |                0.45 |        119855 |                  5000 |               458 |                    30 |                     42 |              0.4  |            0.603424 |               0.746963 |                        0.94 |
| all-MiniLM-L6-v2       |                0.5  |        119855 |                  5000 |               331 |                    30 |                     42 |              0.4  |            0.603424 |               0.746963 |                        0.94 |
| all-MiniLM-L6-v2       |                0.55 |        119855 |                  5000 |               236 |                    30 |                     42 |              0.4  |            0.603424 |               0.746963 |                        0.94 |
| all-MiniLM-L6-v2       |                0.6  |        119855 |                  5000 |               151 |                    30 |                     42 |              0.4  |            0.603424 |               0.746963 |                        0.94 |
| Qwen/Qwen3-Embedding-0.6B |             0.6 |        119855 |                  5000 |               736 |                    43 |                     42 |              0.14 |            0.6361   |               0.743189 |                        0.62 |

## Qwen3 补充验证（同口径，阈值 0.6）

- 本次补充严格沿用 round2 口径：`max_candidates=120000`、`lexical_topn=5000`、`hybrid_alpha=0.75`、时间窗不变。
- 仅替换 embedding 模型为 `Qwen/Qwen3-Embedding-0.6B`，并固定 `min_embedding_sim=0.6`。
- 结果：`semantic_kept_n=736`，说明在该 query 上，Qwen3 在 0.6 下并非“查不到”，而是明显高于 BGE(408) 与 MiniLM(151) 的保留规模。
- 质量侧信号：Qwen3 的 `top50_dup_ratio=0.14` 较低，但 `top50_heuristic_precision=0.62` 低于 MiniLM(0.94)，说明其召回更宽、覆盖更高，同时会引入更多非“trump+china+visit 三词全命中”的语义近邻。

## 观察

- 提高阈值会减少 semantic_kept_n，提高结果纯度，但可能损失覆盖。
- BGE 在中高阈值下通常保持更平衡的相关性与去重表现。
- MiniLM 在本 query 上对关键词更敏感，heuristic 指标可能更高，但重复风险仍需人工复核。
- Qwen3 在 0.6 下保留规模最高，适合先做宽召回；若用于高精度训练集，建议叠加实体约束或更强规则过滤。

## 当前最优（按 heuristic_precision 优先，dup_ratio 次优）

- model_name: all-MiniLM-L6-v2
- min_embedding_sim: 0.45
- semantic_kept_n: 458
- top50_heuristic_precision: 0.940
- top50_dup_ratio: 0.400