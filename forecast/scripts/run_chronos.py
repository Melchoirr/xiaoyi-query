import os
# ⚡️ 国内镜像加速 (必须放在最前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
DATA_DIR = "./dataset"
MODEL_ID = "amazon/chronos-2"
SEQ_LEN = 336
BATCH_SIZE = 32
# =======================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    return parser.parse_args()

def get_data(dataset_name, pred_len, flag='test'):
    """
    返回:
        inputs: [N, seq_len, D]  多变量输入
        targets: [N, pred_len, D]  多变量目标
    """
    path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    df = pd.read_csv(path)

    # 7:1:2 划分
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    border1s = [0, num_train - SEQ_LEN, len(df) - num_test - SEQ_LEN]
    border2s = [num_train, num_train + num_vali, len(df)]

    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    # 排除 date 列
    cols = df.columns[1:] if 'date' in df.columns else df.columns

    # StandardScaler (在 train set 上计算)
    train_data = df[cols].iloc[border1s[0]:border2s[0]].values
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)

    data_vals = df[cols].iloc[border1:border2].values
    data_vals_scaled = (data_vals - mean) / std

    # 生成样本: [N, seq_len, D] 和 [N, pred_len, D]
    inputs, targets = [], []
    valid_samples = len(data_vals_scaled) - SEQ_LEN - pred_len + 1

    for i in range(valid_samples):
        inputs.append(data_vals_scaled[i : i + SEQ_LEN])
        targets.append(data_vals_scaled[i + SEQ_LEN : i + SEQ_LEN + pred_len])

    return np.array(inputs), np.array(targets)

def main():
    args = get_args()
    print(f"▶️  Chronos-2: {args.dataset} | Pred: {args.pred_len}")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 1. 加载模型
    print(f"⏳ Loading Chronos-2 pipeline on {device}...")
    from chronos import Chronos2Pipeline
    if device == "mps":
        # device_map 不支持 "mps"，手动 .to(device)
        pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        pipeline.model = pipeline.model.to(device)
    else:
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_ID, device_map=device, torch_dtype=torch.float32,
        )

    # 2. 准备数据 (Chronos-2 原生多变量)
    inputs, targets = get_data(args.dataset, args.pred_len)
    # inputs: [N, seq_len, D] -> [N, D, seq_len]  (Chronos 期望的格式)
    inputs_mv = inputs.transpose(0, 2, 1)
    # targets: [N, pred_len, D] 保持不变，用于计算指标

    print(f"📊 Samples: {inputs_mv.shape[0]}, Vars: {inputs_mv.shape[1]}, Seq: {inputs_mv.shape[2]}")

    preds = []

    # 3. 推理循环
    for i in tqdm(range(0, len(inputs_mv), BATCH_SIZE), ncols=80, ascii=True, desc="Infer"):
        batch = torch.from_numpy(inputs_mv[i : i + BATCH_SIZE]).float()

        with torch.no_grad():
            # 输出: list of [n_variates, n_quantiles, pred_len] tensors
            predictions = pipeline.predict(batch, prediction_length=args.pred_len)

        # stack -> [batch, n_variates, n_quantiles, pred_len]
        stacked = torch.stack(predictions)
        # 取中位数 (第 n_quantiles//2 个分位数)
        median_idx = stacked.shape[2] // 2
        # [batch, n_variates, pred_len] -> [batch, pred_len, n_variates]
        result = stacked[:, :, median_idx, :].cpu().numpy().transpose(0, 2, 1)
        preds.append(result)

    preds = np.concatenate(preds, axis=0)

    # 4. 计算指标
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    print(f"RESULT: Dataset={args.dataset}, Pred={args.pred_len}, MSE={mse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()
