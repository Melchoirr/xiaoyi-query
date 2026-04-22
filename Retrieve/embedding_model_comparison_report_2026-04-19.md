# Embedding 模型对比实验报告

## 实验设置

- query: Will trump visit china
- window_utc: 2026-02-25 00:00:00 to 2026-03-25 23:59:59
- stage1 max_candidates: 120000
- stage2 lexical_topn: 5000
- stage3 hybrid_alpha: 0.75
- stage3 min_embedding_sim: 0.45
- stage3 embedding_batch_size: 128

## 对比结果

| model_name             |   candidate_n |   lexical_prefilter_n |   semantic_kept_n |   top50_unique_titles |   top50_unique_sources |   top50_dup_ratio |   top50_mean_hybrid |   top50_mean_embedding |   top50_heuristic_precision | status   | error   |
|:-----------------------|--------------:|----------------------:|------------------:|----------------------:|-----------------------:|------------------:|--------------------:|-----------------------:|----------------------------:|:---------|:--------|
| all-MiniLM-L6-v2       |        119855 |                  5000 |               457 |                    30 |                     42 |              0.4  |            0.603424 |               0.746963 |                        0.94 | ok       |         |
| intfloat/e5-base-v2    |        119855 |                  5000 |              5000 |                    44 |                     39 |              0.12 |            0.704705 |               0.805859 |                        0.44 | ok       |         |
| BAAI/bge-large-en-v1.5 |        119855 |                  5000 |              3165 |                    42 |                     39 |              0.16 |            0.593006 |               0.690596 |                        0.7  | ok       |         |

## 指标说明

- top50_heuristic_precision: Top50 标题同时包含 trump+china+visit 的比例，仅作弱代理指标。
- top50_dup_ratio: Top50 标题重复占比，越低越好。
- semantic_kept_n: 通过 embedding 阈值后的候选规模。