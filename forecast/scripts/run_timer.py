import os
# ⚡️ 国内镜像加速 (必须放在最前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# ================= 配置 =================
DATA_DIR = "./dataset"
MODEL_ID = "thuml/timer-base-84m"
SEQ_LEN = 512
BATCH_SIZE = 32
# =======================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    return parser.parse_args()

def get_data(dataset_name, pred_len, flag='test'):
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

    cols = df.columns[1:] if 'date' in df.columns else df.columns

    # StandardScaler (在 train set 上计算)
    train_data = df[cols].iloc[border1s[0]:border2s[0]].values
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)

    data_vals = df[cols].iloc[border1:border2].values
    data_vals_scaled = (data_vals - mean) / std

    inputs, targets = [], []
    valid_samples = len(data_vals_scaled) - SEQ_LEN - pred_len + 1

    for i in range(valid_samples):
        inputs.append(data_vals_scaled[i : i + SEQ_LEN])
        targets.append(data_vals_scaled[i + SEQ_LEN : i + SEQ_LEN + pred_len])

    return np.array(inputs), np.array(targets)

def main():
    args = get_args()
    print(f"▶️  Timer-XL: {args.dataset} | Pred: {args.pred_len}")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 1. 加载模型
    print(f"⏳ Loading Timer-XL on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype='auto',
    ).to(device)
    model.eval()

    # 2. 准备数据 (单变量 flatten)
    inputs, targets = get_data(args.dataset, args.pred_len)

    # [N, seq_len, D] -> [N*D, seq_len]
    s, seq, v = inputs.shape
    inputs_flat = inputs.transpose(0, 2, 1).reshape(-1, seq)
    targets_flat = targets.transpose(0, 2, 1).reshape(-1, args.pred_len)

    inputs_tensor = torch.tensor(inputs_flat, dtype=torch.float32).to(device)

    print(f"📊 Samples: {s}, Vars: {v}, Total sequences: {len(inputs_flat)}")

    preds = []

    # 3. 推理循环
    for i in tqdm(range(0, len(inputs_flat), BATCH_SIZE), ncols=80, ascii=True, desc="Infer"):
        batch_in = inputs_tensor[i : i + BATCH_SIZE].to(model.dtype)

        with torch.no_grad():
            # 输出: [batch, input_len + pred_len]
            outputs = model.generate(batch_in, max_new_tokens=args.pred_len)
            # 取最后 pred_len
            forecast = outputs[:, -args.pred_len:]

        preds.append(forecast.float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 4. 计算指标
    mse = np.mean((preds - targets_flat) ** 2)
    mae = np.mean(np.abs(preds - targets_flat))

    print(f"RESULT: Dataset={args.dataset}, Pred={args.pred_len}, MSE={mse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()
