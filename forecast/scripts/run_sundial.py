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
MODEL_ID = "thuml/sundial-base-128m"
SEQ_LEN = 336
BATCH_SIZE = 32
# ⚠️ 官方示例核心参数：采样次数
# 设为 1 速度最快；设为 20 精度更高 (会对这 20 条结果取平均)
NUM_SAMPLES = 20
# =======================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    return parser.parse_args()

def get_data(dataset_name, pred_len, flag='test'):
    """
    Args:
        dataset_name: 数据集名称 (e.g. 'ETTh1', 'weather')
        pred_len: 预测长度
        flag: 数据集类型 ['train', 'test', 'val']，默认为 'test' 以保持和原函数行为近似
    """
    # 1. 读取数据
    path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    df = pd.read_csv(path)

    # 2. 定义划分边界 (核心修改部分)
    base_name = dataset_name.lower()

    # 其他数据集: 7:1:2 划分
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    border1s = [0, num_train - SEQ_LEN, len(df) - num_test - SEQ_LEN]
    border2s = [num_train, num_train + num_vali, len(df)]

    # 3. 确定当前需要的 Slice
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]

    border1 = border1s[set_type]
    border2 = border2s[set_type]

    # 处理列 (排除 date 列)
    cols = df.columns[1:] if 'date' in df.columns else df.columns

    # 4. 归一化 (StandardScaler)
    # 关键：一定要在 Train set 上计算 mean 和 std
    train_border1 = border1s[0]
    train_border2 = border2s[0]
    train_data = df[cols].iloc[train_border1:train_border2].values

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std) # 避免除以0

    # 5. 获取并缩放目标数据
    # 根据 flag 获取对应的数据切片
    data_vals = df[cols].iloc[border1:border2].values
    data_vals_scaled = (data_vals - mean) / std

    # 6. 生成时间序列样本
    inputs, targets = [], []
    # 有效样本数计算：总长度 - 输入序列长 - 预测序列长 + 1
    valid_samples = len(data_vals_scaled) - SEQ_LEN - pred_len + 1

    for i in range(valid_samples):
        inputs.append(data_vals_scaled[i : i + SEQ_LEN])
        targets.append(data_vals_scaled[i + SEQ_LEN : i + SEQ_LEN + pred_len])

    return np.array(inputs), np.array(targets)

def main():
    args = get_args()
    print(f"▶️  Sundial (Samples={NUM_SAMPLES}): {args.dataset} | Pred: {args.pred_len}")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # 1. 加载模型 (使用默认精度，不强制 float16/float32)
    print("⏳ Loading model with default configuration...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # 2. 准备数据
    inputs, targets = get_data(args.dataset, args.pred_len)

    # [Samples, Seq, Vars] -> [N, Seq]
    s, seq, v = inputs.shape
    inputs_flat = inputs.transpose(0, 2, 1).reshape(-1, seq)
    targets_flat = targets.transpose(0, 2, 1).reshape(-1, args.pred_len)

    # 转为 Tensor (默认是 Float32)
    inputs_tensor = torch.tensor(inputs_flat, dtype=torch.float32).to(device)

    preds = []

    # 3. 推理循环
    for i in tqdm(range(0, len(inputs_flat), BATCH_SIZE), ncols=80, mininterval=1.0, ascii=True, desc="Infer"):
        batch_in = inputs_tensor[i : i + BATCH_SIZE]

        # 🔥 关键修复：自动对齐类型 🔥
        batch_in = batch_in.to(model.dtype)

        with torch.no_grad():
            forecast_raw = model.generate(
                batch_in,
                max_new_tokens=args.pred_len,
                num_samples=NUM_SAMPLES
            )

            # 处理输出维度
            if forecast_raw.shape[-1] > args.pred_len:
                forecast_slice = forecast_raw[..., -args.pred_len:]
            else:
                forecast_slice = forecast_raw

            # 多样本取平均 (Ensemble)
            if forecast_slice.ndim == 3 and forecast_slice.shape[1] == NUM_SAMPLES:
                point_forecast = forecast_slice.mean(dim=1)
            else:
                point_forecast = forecast_slice

            preds.append(point_forecast.float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 4. 计算指标
    mse = np.mean((preds - targets_flat) ** 2)
    mae = np.mean(np.abs(preds - targets_flat))

    print(f"RESULT: Dataset={args.dataset}, Pred={args.pred_len}, MSE={mse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()
