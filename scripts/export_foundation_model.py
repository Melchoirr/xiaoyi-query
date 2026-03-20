"""
scripts/export_foundation_model.py
===================================
离线脚本：将预训练时序基础模型 TS2Vec 的表征编码器导出为 ONNX。

【模型选型依据】
-----------------
我们选择 TS2Vec（Time Series to Vector）作为特征编码器，原因如下：

1. **学术背景**
   TS2Vec（Yue et al., 2022, "TS2Vec: Towards Universal Representation of Time Series")
   是时序表征学习的里程碑工作，提出了层次化对比学习框架，在 UCR/UEA/TSRR 等
   多个基准上取得了 SOTA 的 Zero-shot 表征迁移性能。

2. **架构适合 RAG 检索**
   TS2Vec 为输入序列的每个时间步生成上下文表征（Contextual Representation），
   对时间维度做 Mean Pooling 后得到全局序列向量（Global Representation），
   非常适合作为语义检索的 Query/Document Embedding。

3. **动态输入长度 + 固定输出维度**
   TS2Vec Encoder 接受任意长度的 1D 序列，通过 Zero-padding + Masking
   实现变长输入，输出固定 320 维稠密向量，完全兼容 Qdrant 的固定向量维度要求。

4. **纯 PyTorch，无自定义算子**
   模型由标准 nn.Linear / LayerNorm / GELU 构成，可直接导出为 ONNX，
   不依赖 TorchScript 以外的任何特殊后端，onnxruntime 可原生推理。

5. **跨域 Zero-shot 能力**
   论文在 128 个数据集上预训练，涵盖医疗、能源、传感器等多元领域，
   模型已学习到时序的模式共性，ETT 数据无需微调即可直接使用。

【输入 / 输出张量规格】
-------------------------
ONNX 输入:
    name:  "input"
    shape: [batch_size, input_len, 1]
            batch_size:  推理时固定为 1（逐条编码）
            input_len:   编码器期望的固定序列长度（默认 512，低于此长度需 Zero-padding）
            1:           通道数（单变量时序，仅支持 univarite）

    注意: 低于 512 长的序列会在前端补 0（Zero-padding）；超过 512 的序列会截断。

ONNX 输出:
    name:  "embedding"
    shape: [batch_size, 1, embedding_dim]
            embedding_dim: 320（TS2Vec base 模型隐层维度）

    输出后在线推理层（encoder.py）执行：
        1. squeeze([0, 0])  → [embedding_dim] 即 [320]
        2. L2 Normalize      → 归一化到单位球面（适配 Qdrant Cosine 距离）

【使用方式】
--------------
# 步骤 1: 在有 torch + ts2vec 环境的机器上运行（可 GPU 可 CPU）
python scripts/export_foundation_model.py

# 步骤 2: 将生成的 models/foundation_encoder.onnx 拷贝到目标机器
# 步骤 3: 重启 FastAPI 服务（自动加载新的 ONNX 模型）
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 依赖说明
# ---------------------------------------------------------------------------
# 此脚本运行在独立导出环境中，需要以下依赖（不需要在线推理环境安装）：
#   pip install ts2vec torch onnxscript
#
# 安装方式（推荐使用独立的 conda/venv）：
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install ts2vec onnxscript
#
# 此脚本不依赖 onnxruntime（仅在线推理侧使用）。
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    print("错误: 此脚本需要在有 torch 的环境中运行。")
    print("建议创建独立 conda 环境：conda create -n ts2vec_export python=3.11")
    print("然后 pip install torch ts2vec onnxscript")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 模型下载与加载
# ---------------------------------------------------------------------------

def load_ts2vec_encoder(
    repo_version: str = "1.0.0",
    device: str = "cpu",
) -> torch.nn.Module:
    """
    下载并返回 TS2Vec 表征编码器（剥离预测头）。

    参数:
        repo_version: ts2vec 包版本（默认 "1.0.0"）
        device: "cpu" 或 "cuda"

    返回:
        TS2Vec Encoder 的 torch.nn.Module，forward(x) → [B, T, d]
        其中 d = 320（隐层维度）
    """
    from ts2vec import TS2Vec

    print(f"[TS2Vec] 正在加载预训练模型 (device={device})...")

    model = TS2Vec(
        input_dims=1,         # TS2Vec 默认：每个时间步 1 个特征（univariate）
        device=device,
    )

    # TS2Vec._net 是 TSEncoder（torch.nn.Module），forward(x: Tensor) → [B, T, d]
    # 不能用 model.encode，它只是对 _net 的一层 numpy 封装
    encoder = model._net
    encoder.eval()
    encoder.to(device)

    # 验证 encoder 输出形状（直接用 torch tensor 调用 _net）
    with torch.no_grad():
        test_input = torch.randn(2, 512, 1)  # [B=2, T=512, C=1]
        test_output = encoder(test_input)
        assert test_output.shape == (2, 512, 320), (
            f"TS2Vec Encoder 输出形状应为 [batch, 512, 320]，"
            f"实际 {test_output.shape}"
        )

    print(f"[TS2Vec] Encoder 加载成功，输出维度: {test_output.shape}")
    return encoder


# ---------------------------------------------------------------------------
# ONNX 导出
# ---------------------------------------------------------------------------

def export_to_onnx(
    encoder: torch.nn.Module,
    output_path: str,
    input_length: int = 512,
    opset_version: int = 18,
) -> None:
    """
    将 TS2Vec Encoder 导出为 ONNX 模型。

    参数:
        encoder:       TS2Vec encoder 模块
        output_path:   ONNX 文件保存路径（含文件名）
        input_length:  固定序列长度（默认 512）
        opset_version: ONNX opset 版本（默认 18，兼容 onnxruntime >= 1.16）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"[导出] ONNX 文件已存在: {output_path}")
        print("[导出] 跳过重新导出。如需重新导出，请先删除该文件。")
        return

    print(f"[导出] 正在导出 ONNX 模型...")
    print(f"       输出路径: {output_path}")
    print(f"       输入形状: [batch=1, seq_len={input_length}, channels=1]")
    print(f"       输出形状: [batch=1, seq_len={input_length}, embedding_dim=320]")
    print(f"       Opset:    {opset_version}")

    encoder.eval()

    # -------------------------------------------------------------------------
    # Dummy Input 张量规格（必须与在线推理侧一致）：
    #
    #   dummy_input.shape = [batch_size, input_length, num_features]
    #
    #   batch_size    = 1       （推理时逐条编码，batch 维度固定为 1）
    #   input_length  = 512     （TS2Vec 内部 patch 大小为 2，
    #                              编码器期望 2 的幂次长度；低于此长度需 Zero-padding）
    #   num_features  = 1       （单变量时序，仅含数值列的值）
    #
    # Zero-padding 策略（在线推理侧 encoder.py 实现）：
    #   - 实际序列长度 L < input_length：前端补 (input_length - L) 个 0
    #   - 实际序列长度 L > input_length：截取最后 input_length 个点
    # -------------------------------------------------------------------------
    dummy_input = torch.randn(1, input_length, 1)

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                # batch 维度固定为 1，不需要动态轴
                # sequence 长度固定为 input_length（输入侧已做 padding）
            },
            opset_version=opset_version,
            external_data=False,
        )

    # -------------------------------------------------------------------------
    # 验证 ONNX 文件
    # -------------------------------------------------------------------------
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[导出] ONNX 模型验证通过: {output_path}")

    # 打印输入输出 shape
    onnx_inputs = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                    for i in onnx_model.graph.input}
    onnx_outputs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim]
                     for o in onnx_model.graph.output}
    print(f"[导出] ONNX 输入:  {onnx_inputs}")
    print(f"[导出] ONNX 输出:  {onnx_outputs}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TS2Vec Encoder ONNX 导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="ONNX 模型输出目录（默认: models/）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="foundation_encoder.onnx",
        help="ONNX 模型文件名（默认: foundation_encoder.onnx）",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=512,
        dest="input_length",
        help="TS2Vec 期望的固定序列长度（默认 512，低于此长度需 Zero-padding）",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        dest="opset",
        help="ONNX opset 版本（默认 18）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="加载模型所使用的设备（默认 cpu）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新导出（覆盖已存在的 ONNX 文件）",
    )
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, args.output_name)

    # 强制重新导出时删除旧文件
    if args.force and os.path.exists(output_path):
        os.remove(output_path)
        print(f"[导出] 已删除旧文件: {output_path}")

    # Step 1: 下载并加载 TS2Vec 编码器
    encoder = load_ts2vec_encoder(device=args.device)

    # Step 2: 导出为 ONNX
    export_to_onnx(
        encoder=encoder,
        output_path=output_path,
        input_length=args.input_length,
        opset_version=args.opset,
    )

    # Step 3: 打印使用说明
    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)
    print(f"ONNX 模型路径: {output_path}")
    print(f"输入规格:  [batch=1, seq_len={args.input_length}, channels=1]")
    print(f"输出规格:  [batch=1, seq_len={args.input_length}, embedding_dim=320]")
    print()
    print("【下一步】")
    print(f"  1. 将 {output_path} 拷贝到目标机器的相同路径")
    print("  2. 更新 components/encoder.py 中的 ONNX_PATH 和 embedding_dim")
    print("  3. 清空并重新摄入数据: rm -rf qdrant_data/")
    print(f"  4. 运行: python scripts/ingest_ett.py --file ETTh1 --column OT")
    print("  5. 启动服务: uvicorn main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
