#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Benchmark - 统一基准测试入口
支持模型: sundial, timemoe, moirai, timerxl, chronos, tirex
支持 train/val/test 三种 flag（不打乱，保持顺序）
数据划分使用 ETT 原始固定边界
输出格式兼容 forecast/fusion/stacking.py
"""

import os
import sys

# ⚡️ 国内镜像加速 (必须放在最前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add local packages to path
TIREX_PATH = os.path.join(os.path.dirname(__file__), '..', 'tirex', 'src')
if TIREX_PATH not in sys.path:
    sys.path.insert(0, TIREX_PATH)

CHRONOS_PATH = os.path.join(os.path.dirname(__file__), '..', 'chronos-forecasting', 'src')
if CHRONOS_PATH not in sys.path:
    sys.path.insert(0, CHRONOS_PATH)

import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ================= 模型配置 =================
MODEL_CONFIG = {
    'sundial': {
        'model_id': 'thuml/sundial-base-128m',
        'num_samples': 20,
    },
    'timemoe': {
        'model_id': 'Maple728/TimeMoE-200M',
        'num_samples': 1,
    },
    'moirai': {
        'model_id': 'Salesforce/moirai-2.0-R-small',
        'num_samples': 1,
    },
    'timerxl': {
        'model_id': 'thuml/timer-base-84m',
        'num_samples': 1,
    },
    'chronos': {
        'model_id': 'amazon/chronos-2',
        'num_samples': 1,
    },
    'tirex': {
        'model_id': 'NX-AI/TiRex',
        'num_samples': 1,
    },
}

MOIRAI_SIZE_MAP = {
    'small': 'Salesforce/moirai-2.0-R-small',
    'base': 'Salesforce/moirai-2.0-R-base',
    'large': 'Salesforce/moirai-2.0-R-large',
}

# 模型名映射 (benchmark 小写 -> setting 中的名字)
MODEL_NAME_MAP = {
    'sundial': 'Sundial',
    'timemoe': 'TimeMoE',
    'moirai': 'Moirai',
    'timerxl': 'Timer',
    'chronos': 'Chronos',
    'tirex': 'TiRex',
}
# ============================================


# ================= 数据加载 =================

# ETT 原始固定边界
ETT_BORDERS = {
    'ETTh1': {
        'border1s': lambda seq_len: [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len],
        'border2s': [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24],
        'freq': 'h',
    },
    'ETTh2': {
        'border1s': lambda seq_len: [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len],
        'border2s': [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24],
        'freq': 'h',
    },
    'ETTm1': {
        'border1s': lambda seq_len: [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len],
        'border2s': [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4],
        'freq': 't',
    },
    'ETTm2': {
        'border1s': lambda seq_len: [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len],
        'border2s': [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4],
        'freq': 't',
    },
}


def load_data(dataset_name, pred_len, seq_len, data_dir, flag='test'):
    """
    使用 ETT 原始固定边界加载数据。

    Returns:
        inputs: [N, seq_len, D]
        targets: [N, pred_len, D]
    """
    path = os.path.join(data_dir, f"{dataset_name}.csv")
    df = pd.read_csv(path)

    # 获取边界
    if dataset_name in ETT_BORDERS:
        borders = ETT_BORDERS[dataset_name]
        border1s = borders['border1s'](seq_len)
        border2s = borders['border2s']
    else:
        # 非 ETT 数据集使用 7:1:2 比例
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_vali = len(df) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df)]

    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    # 排除 date 列
    cols = df.columns[1:] if 'date' in df.columns else df.columns

    # StandardScaler (在 train set 上计算)
    scaler = StandardScaler()
    train_data = df[cols].iloc[border1s[0]:border2s[0]].values
    scaler.fit(train_data)

    # 对全量数据做 transform，然后切片
    all_data = scaler.transform(df[cols].values)
    data_slice = all_data[border1:border2]

    # 生成样本
    inputs, targets = [], []
    valid_samples = len(data_slice) - seq_len - pred_len + 1

    for i in range(valid_samples):
        inputs.append(data_slice[i : i + seq_len])
        targets.append(data_slice[i + seq_len : i + seq_len + pred_len])

    return np.array(inputs), np.array(targets)


def flatten_multivariate(inputs, targets):
    """[N, seq_len, D] -> [N*D, seq_len], [N, pred_len, D] -> [N*D, pred_len]"""
    _, seq, _ = inputs.shape
    inputs_flat = inputs.transpose(0, 2, 1).reshape(-1, seq)
    targets_flat = targets.transpose(0, 2, 1).reshape(-1, targets.shape[1])
    return inputs_flat, targets_flat


def unflatten_predictions(preds_flat, n_samples, n_vars, pred_len):
    """[N*D, pred_len] -> [N, pred_len, D]"""
    # [N*D, pred_len] -> [N, D, pred_len] -> [N, pred_len, D]
    return preds_flat.reshape(n_samples, n_vars, pred_len).transpose(0, 2, 1)


# ================= 模型加载 =================

def load_model(model_type, model_id, pred_len, seq_len, device):
    """加载模型"""

    if model_type == 'sundial':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype='auto',
        ).to(device)
        model.eval()
        return model

    elif model_type == 'timemoe':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device,
            torch_dtype='auto', trust_remote_code=True
        )
        model.eval()
        return model

    elif model_type == 'moirai':
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(model_id),
            prediction_length=pred_len,
            context_length=seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)
        model.eval()
        return model

    elif model_type == 'timerxl':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype='auto',
        ).to(device)
        model.eval()
        return model

    elif model_type == 'chronos':
        from chronos import Chronos2Pipeline
        if device == 'mps':
            pipeline = Chronos2Pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
            pipeline.model = pipeline.model.to(device)
        else:
            pipeline = Chronos2Pipeline.from_pretrained(
                model_id, device_map=device, torch_dtype=torch.float32,
            )
        return pipeline

    elif model_type == 'tirex':
        from tirex import load_model as tirex_load_model
        model = tirex_load_model(
            model_id, device=device, backend='torch', compile=False,
        )
        return model


# ================= 推理 =================

def inference_batch(model, model_type, batch, pred_len, num_samples, device):
    """单批次推理"""

    if model_type == 'sundial':
        batch_tensor = torch.from_numpy(batch).to(device=device, dtype=model.dtype)
        with torch.no_grad():
            forecast = model.generate(batch_tensor, max_new_tokens=pred_len, num_samples=num_samples)
            forecast = forecast[..., -pred_len:].mean(dim=1) if forecast.ndim == 3 else forecast[..., -pred_len:]
        return forecast.float().cpu().numpy()

    elif model_type == 'timemoe':
        batch_tensor = torch.from_numpy(batch).to(device=device, dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(inputs=batch_tensor, max_new_tokens=pred_len)
        return outputs[:, -pred_len:].float().cpu().numpy()

    elif model_type == 'moirai':
        past_target = [arr[:, np.newaxis] for arr in batch]
        predictions = model.predict(past_target=past_target)
        median_idx = predictions.shape[1] // 2
        return predictions[:, median_idx, :]

    elif model_type == 'timerxl':
        batch_tensor = torch.from_numpy(batch).to(device=device, dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(batch_tensor, max_new_tokens=pred_len)
        return outputs[:, -pred_len:].float().cpu().numpy()

    elif model_type == 'chronos':
        batch_tensor = torch.from_numpy(batch).float()
        predictions = model.predict(batch_tensor, prediction_length=pred_len)
        stacked = torch.stack(predictions)
        median_idx = stacked.shape[2] // 2
        result = stacked[:, :, median_idx, :].cpu().numpy()
        return result.transpose(0, 2, 1)  # [batch, pred_len, n_variates]

    elif model_type == 'tirex':
        batch_tensor = torch.from_numpy(batch).float()
        quantiles, mean = model.forecast(context=batch_tensor, prediction_length=pred_len)
        if isinstance(mean, torch.Tensor):
            return mean.cpu().numpy()
        return mean


# ================= 主函数 =================

def get_args():
    parser = argparse.ArgumentParser(description="Unified Benchmark")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()),
                        help='Model type')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, ...)')
    parser.add_argument('--pred_len', type=int, required=True,
                        help='Prediction length')
    parser.add_argument('--flags', type=str, default='test',
                        help='Comma-separated flags to run: test,val,train')
    parser.add_argument('--model_id', type=str, default='',
                        help='Override default model ID')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size (moirai only)')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Num samples (0=use default)')
    parser.add_argument('--features', type=str, default='M',
                        choices=['M', 'S', 'MS'])
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--result_path', type=str, default='./outputs/results/')
    return parser.parse_args()


def run_one_flag(model, model_type, args, model_id, num_samples, flag, device):
    """对单个 flag (train/val/test) 执行推理并保存结果"""
    print(f"\n{'='*60}")
    print(f"  {flag.upper()} set inference")
    print(f"{'='*60}")

    # 1. 加载数据
    inputs, targets = load_data(args.dataset, args.pred_len, args.seq_len, args.data_dir, flag=flag)
    n_samples, seq_len, n_vars = inputs.shape
    print(f"Data: {n_samples} samples, {n_vars} vars, seq={seq_len}")

    # 2. 准备数据（按模型类型）
    if model_type == 'chronos':
        # 多变量模式: [N, seq_len, D] -> [N, D, seq_len]
        inputs_data = inputs.transpose(0, 2, 1)
        targets_data = targets  # [N, pred_len, D]
        is_multivariate = True
    else:
        # 单变量 flatten: [N, seq_len, D] -> [N*D, seq_len]
        inputs_data, targets_flat = flatten_multivariate(inputs, targets)
        targets_data = targets_flat
        is_multivariate = False

    # 3. 推理
    preds = []
    for i in tqdm(range(0, len(inputs_data), args.batch_size),
                  desc=f"{flag}", ncols=80, ascii=True):
        batch = inputs_data[i:i + args.batch_size]
        pred = inference_batch(model, model_type, batch, args.pred_len, num_samples, device)
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    # 4. 还原形状为 [N, pred_len, D]
    if is_multivariate:
        # chronos 已经是 [N, pred_len, D]
        preds_3d = preds
        trues_3d = targets_data
    else:
        # [N*D, pred_len] -> [N, pred_len, D]
        preds_3d = unflatten_predictions(preds, n_samples, n_vars, args.pred_len)
        trues_3d = targets  # 原始的 [N, pred_len, D]

    # 5. 计算指标（在归一化数据上）
    from forecast.utils.metrics import metric as calc_metric
    mae, mse, rmse, mape, mspe = calc_metric(preds_3d, trues_3d)

    # 6. 构建 setting 名并保存
    model_display = MODEL_NAME_MAP.get(model_type, model_type)
    setting = f"{model_display}_{args.dataset}_{args.features}_sl{args.seq_len}_pl{args.pred_len}"

    if flag in ('val', 'train'):
        save_dir = os.path.join(args.result_path, setting, flag)
    else:
        save_dir = os.path.join(args.result_path, setting)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'pred.npy'), preds_3d)
    np.save(os.path.join(save_dir, 'true.npy'), trues_3d)
    np.save(os.path.join(save_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

    print(f"  MSE={mse:.4f}, MAE={mae:.4f}")
    print(f"  Saved: {save_dir}/pred.npy  shape={preds_3d.shape}")
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"RESULT|{ts}|{setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")

    return mse, mae


def main():
    args = get_args()

    # 解析模型配置
    model_config = MODEL_CONFIG[args.model]
    model_id = args.model_id or (
        MOIRAI_SIZE_MAP.get(args.model_size) if args.model == 'moirai'
        else model_config['model_id']
    )
    num_samples = args.num_samples if args.num_samples > 0 else model_config['num_samples']
    flags = [f.strip() for f in args.flags.split(',')]

    # 设备
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Model: {args.model} ({model_id})")
    print(f"Dataset: {args.dataset}, Pred: {args.pred_len}, Seq: {args.seq_len}")
    print(f"Device: {device}, Batch: {args.batch_size}, Samples: {num_samples}")
    print(f"Flags: {flags}")

    # 加载模型（只加载一次）
    t0 = time.time()
    model = load_model(args.model, model_id, args.pred_len, args.seq_len, device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # 对每个 flag 执行推理
    for flag in flags:
        run_one_flag(model, args.model, args, model_id, num_samples, flag, device)

    print("\nDone!")


if __name__ == "__main__":
    main()
