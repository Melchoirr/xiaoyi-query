import csv
import json
import os
import subprocess
import sys

PY = r'.\\.venv\\Scripts\\python.exe'
BASE = [
    'run.py',
    '--model', 'PatternSearch',
    '--data', 'ETTh1',
    '--root_path', './dataset/',
    '--data_path', 'ETTh1.csv',
    '--seq_len', '96',
    '--pred_len', '96',
    '--batch_size', '256',
    '--top_k', '5',
    '--result_path', './results_round3',
]

EXPS = [
    ('A1_target_top1', ['--distance_mode','target_only','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','top1','--rerank_mode','none']),
    ('A2_target_inv', ['--distance_mode','target_only','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','inverse_distance','--rerank_mode','none']),
    ('A3_target_softmax', ['--distance_mode','target_only','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','none']),
    ('A4_weight_top1', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','top1','--rerank_mode','none']),
    ('A5_weight_inv', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','inverse_distance','--rerank_mode','none']),
    ('A6_weight_softmax', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','none']),
    ('A7_weight_softmax_hybrid', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','hybrid']),

    ('H1_shape_only', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','hybrid','--rerank_alpha','1.0','--rerank_beta','0.0','--rerank_gamma','0.0','--rerank_delta','0.0']),
    ('H2_shape_level', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','hybrid','--rerank_alpha','1.0','--rerank_beta','0.5','--rerank_gamma','0.0','--rerank_delta','0.0']),
    ('H3_shape_level_scale', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','hybrid','--rerank_alpha','1.0','--rerank_beta','0.5','--rerank_gamma','0.3','--rerank_delta','0.0']),
    ('H4_shape_level_scale_phase', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','hybrid','--rerank_alpha','1.0','--rerank_beta','0.5','--rerank_gamma','0.3','--rerank_delta','0.2']),

    ('G1_mean', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','mean','--rerank_mode','none']),
    ('G2_top1', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','top1','--rerank_mode','none']),
    ('G3_inverse_distance', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','inverse_distance','--rerank_mode','none']),
    ('G4_softmax_t0.2', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','0.2','--rerank_mode','none']),
    ('G5_softmax_t0.5', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','0.5','--rerank_mode','none']),
    ('G6_softmax_t1.0', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','1.0','--rerank_mode','none']),
    ('G7_softmax_t2.0', ['--distance_mode','weighted_channel','--future_representation','relative_norm','--restoration_mode','auto','--aggregation_mode','softmax_temp','--aggregation_temperature','2.0','--rerank_mode','none']),
]

rows = []
for name, extra in EXPS:
    cmd = [PY] + BASE + extra
    print('RUN', name)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(f'failed: {name}')

    # read last appended row from summary
    summary = os.path.join('results_round3','summary.csv')
    with open(summary, 'r', encoding='utf-8') as f:
        last = list(csv.DictReader(f))[-1]

    setting = f"PatternSearch_ETTh1_sl96_pl96_k5_futrelative_norm_dist{extra[1] if '--distance_mode' in extra else 'weighted_channel'}_agg{extra[extra.index('--aggregation_mode')+1]}_rer{extra[extra.index('--rerank_mode')+1]}"

    # find diagnostics via latest run dir by mtime
    root = 'results_round3'
    dirs = [os.path.join(root,d) for d in os.listdir(root) if os.path.isdir(os.path.join(root,d)) and d.startswith('PatternSearch_ETTh1_')]
    dirs.sort(key=lambda x: os.path.getmtime(x))
    latest = dirs[-1]
    diag_path = os.path.join(latest, 'reports', 'diagnostics.json')
    diag = {}
    if os.path.exists(diag_path):
        with open(diag_path, 'r', encoding='utf-8') as f:
            diag = json.load(f)

    rows.append({
        'exp': name,
        'mae': float(last['mae']),
        'mse': float(last['mse']),
        'rmse': float(last['rmse']),
        'mape': float(last['mape']) if last['mape'] else None,
        'smape': float(last['smape']) if last['smape'] else None,
        'runtime': float(last['runtime']),
        'cand_changed': diag.get('candidate_changed_ratio'),
        'pre_tdist': diag.get('pre_target_dist_mean'),
        'post_tdist': diag.get('post_target_dist_mean'),
        'tdist_improve': diag.get('target_dist_improvement'),
        'w_entropy': diag.get('weight_entropy_mean'),
        'w_max': diag.get('weight_max_mean'),
        'rer_shape': diag.get('rerank_shape_mean'),
        'rer_level': diag.get('rerank_level_mean'),
        'rer_scale': diag.get('rerank_scale_mean'),
        'rer_phase': diag.get('rerank_phase_mean'),
    })

out_csv = os.path.join('results_round3','pattern_round3_ablation.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print('saved', out_csv)
