# PrimitiveFusion 模型设计文档

## 1. 动机

多变量时序预测中，Channel-Independent (CI) 方法（如 PatchTST）忽略了通道间的信息，而传统 Channel-Dependent (CD) 方法直接对所有通道做交叉注意力，面临 O(C^2) 的复杂度和噪声干扰问题。

PrimitiveFusion 提出一种折中方案：将非目标通道的序列通过可学习的"原语码本"压缩为少量语义 token，再通过 cross-attention 注入目标通道的预测路径。这实现了：

- **信息瓶颈**：C-1 个通道 → K 个原语 token（K 固定，与通道数无关）
- **软聚类**：行为相似的通道自动聚合到同一原语
- **选择性融合**：目标通道通过 attention 机制选择性地关注有用的原语

灵感来源：TEST 论文的 text prototype 思想（用文本原型对齐时序嵌入），以及 VQ-VAE 的码本机制。

## 2. 架构总览

```
Input x: [B, L, C]
    │
    ├── 目标通道 x_t: [B, L, 1]
    │       │
    │       ▼
    │   ┌──────────────────┐
    │   │  PatchEmbedding   │  切片 → 线性映射 → 位置编码
    │   └──────────────────┘
    │       │ [B, N, d_model]
    │       ▼
    │   ┌──────────────────┐
    │   │ Transformer Enc   │  e_layers 层 self-attention
    │   │ (与 PatchTST 相同) │
    │   └──────────────────┘
    │       │ target_repr: [B, N, d_model]
    │       │
    │       │         ┌─────────────────────────────────────────┐
    │       │         │          其他通道路径                       │
    │       │         │                                           │
    │       │         │  x_o: [B, L, C-1]                        │
    │       │         │       │                                   │
    │       │         │       ▼                                   │
    │       │         │  ┌────────────────┐                      │
    │       │         │  │ ChannelEncoder  │ Linear(L → d_model) │
    │       │         │  └────────────────┘                      │
    │       │         │       │ channel_repr: [B, C-1, d_model]  │
    │       │         │       ▼                                   │
    │       │         │  ┌─────────────────────┐                 │
    │       │         │  │ PrimitiveAssignment   │                │
    │       │         │  │  codebook: [K, d_model]│               │
    │       │         │  │  sim → softmax → 聚合   │              │
    │       │         │  └─────────────────────┘                 │
    │       │         │       │ primitive_tokens: [B, K, d_model]│
    │       │         └───────┼──────────────────────────────────┘
    │       │                 │
    │       ▼                 ▼
    │   ┌──────────────────────────┐
    │   │    Cross-Attention        │
    │   │  Q = target_repr          │
    │   │  K, V = primitive_tokens  │
    │   └──────────────────────────┘
    │       │ fused: [B, N, d_model]
    │       ▼
    │   Flatten → Linear Head → pred_c: [B, pred_len]
    │
    └── 对所有 C 个通道循环执行上述过程
            │
            ▼
        Stack → x_out: [B, pred_len, C]
```

## 3. 核心组件

### 3.1 ChannelEncoder

将每个非目标通道的完整时间序列编码为单个 d_model 维向量。

```
输入: [B, C-1, L]（C-1 个通道，每个长度 L）
      │
      ▼
  Linear(L, d_model) ──→ Dropout ──→ LayerNorm
      │
输出: [B, C-1, d_model]（每个通道一个向量）
```

设计选择：使用单层线性映射而非 CNN/RNN，因为：
- 简单高效，适合原型阶段
- 每个通道独立编码，不引入通道间的隐式交互

### 3.2 PrimitiveAssignment（原语分配）

核心创新模块。维护一个可学习的原语码本 `codebook: [K, d_model]`，通过 soft attention 将 C-1 个通道表征聚合为 K 个原语 token。

**计算过程**：

```
Step 1: 计算相似度
  sim = channel_repr @ codebook^T / √d_model    → [B, C-1, K]
  含义：每个通道与每个原语的匹配程度

Step 2: 软分配
  weights = softmax(sim / τ, dim=-1)             → [B, C-1, K]
  含义：每个通道对 K 个原语的分配权重（每行和为 1）
  τ 为温度参数，τ 越小分配越集中

Step 3: 加权聚合
  primitive_tokens = weights^T @ channel_repr    → [B, K, d_model]
  含义：每个原语 token = 所有通道表征的加权平均
        权重取决于通道与该原语的相似度
```

**直觉**：这是一个软聚类过程。K 个原语相当于 K 个聚类中心，每个原语"吸收"与自己相似的通道的信息。例如，如果温度和湿度都与原语 P3 高度相似，那么 P3 就会成为"气象类"通道的聚合表示。

**参数**：
- `num_primitives (K)`：原语数量，默认 16
- `temperature (τ)`：softmax 温度，默认 1.0

### 3.3 CrossAttentionLayer

标准的 cross-attention 层，让目标通道的 patch 表征选择性地关注原语信息。

```
输入:
  target_repr:      [B, N_patches, d_model]  ← Query
  primitive_tokens: [B, K, d_model]          ← Key, Value

计算:
  1. Pre-norm: x = LayerNorm(target_repr)
  2. Cross-Attention: attn_out = MultiHeadAttn(Q=x, K=primitives, V=primitives)
  3. 残差连接: x = target_repr + attn_out
  4. Pre-norm FFN: x = x + FFN(LayerNorm(x))

输出: [B, N_patches, d_model]
```

FFN 结构：`Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model) → Dropout`

### 3.4 目标通道路径

与 PatchTST 完全一致的结构，保证可以直接做消融对比：

1. **PatchEmbedding**：将长度 L 的单通道序列切成 N 个 patch
   - Padding → Unfold → Linear(patch_len, d_model) → PositionalEncoding
   - `N = ⌊(L - patch_len + padding) / stride⌋ + 1`

2. **Transformer Encoder**：e_layers 层标准 TransformerEncoderLayer
   - Self-Attention + FFN，GELU 激活，batch_first=True

3. **Prediction Head**：`Flatten → Dropout → Linear(N × d_model, pred_len)`

### 3.5 RevIN

可选的 Reversible Instance Normalization：
- 前向：对输入逐实例归一化（减均值、除标准差）
- 预测后：反归一化（乘标准差、加均值）
- 缓解训练/测试分布偏移

## 4. 前向传播流程

```python
def forward(x):  # x: [B, L, C]
    if use_revin: x = revin.normalize(x)

    for c in range(C):
        # 1. 提取目标通道
        x_t = x[:, :, c:c+1]                    # [B, L, 1]

        # 2. 目标通道：patch + self-attention
        x_patch = patch_embedding(x_t)           # [B, N, d_model]
        target_repr = encoder(x_patch)           # [B, N, d_model]

        # 3. 其他通道 → 原语
        x_o = x[:, :, other_channels]            # [B, L, C-1]
        channel_repr = channel_encoder(x_o)      # [B, C-1, d_model]
        primitives = primitive_assignment(channel_repr)  # [B, K, d_model]

        # 4. Cross-attention 融合
        fused = cross_attention(target_repr, primitives)  # [B, N, d_model]

        # 5. 预测
        pred_c = head(flatten(fused))            # [B, pred_len]

    x_out = stack(all_pred_c)                    # [B, pred_len, C]
    if use_revin: x_out = revin.denormalize(x_out)
    return x_out
```

关键特点：**逐通道循环**——每个通道轮流作为目标，其余通道提供跨通道上下文。所有通道共享同一套编码器、码本和预测头。

## 5. 超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `seq_len` | 512 | 输入历史长度 |
| `pred_len` | 96 | 预测长度 |
| `enc_in` | 7 | 输入通道数 |
| `d_model` | 128 | 嵌入维度 |
| `n_heads` | 8 | 注意力头数 |
| `e_layers` | 3 | Transformer 编码器层数 |
| `d_ff` | 256 | FFN 隐藏层维度 |
| `patch_len` | 16 | patch 长度 |
| `stride` | 8 | patch 步长 |
| `dropout` | 0.1 | Dropout 率 |
| `num_primitives` | 16 | 原语码本大小 K |
| `primitive_temp` | 1.0 | 软分配温度 τ |
| `n_cross_layers` | 1 | Cross-attention 层数 |
| `revin` | False | 是否启用 RevIN |

## 6. 模型规模

以默认配置（ETTh1, 7 通道）为例：

- **总参数量**：~1.39M
- **模型大小**：~5.3 MB
- **单 epoch 训练时间**：~40s (MPS), ~128s (CPU)

## 7. 与相关方法的对比

| 方法 | 跨通道方式 | 通道间交互 | 信息压缩 |
|------|-----------|-----------|---------|
| PatchTST | 无 (CI) | 不建模 | — |
| iTransformer | 通道做 token | 全 attention O(C^2) | 无 |
| CrossFormer | 跨通道 attention | 全 attention | 无 |
| **PrimitiveFusion** | 原语码本 | 软聚类 + cross-attn | C-1 → K 个原语 |

## 8. 已知问题与改进方向

### 8.1 原语分配均匀化

实验发现：训练后原语分配权重接近均匀分布（entropy ≈ 100%），K 个原语没有学到差异化的角色。

**原因**：codebook 的学习完全依赖 MSE 预测损失的反向传播，梯度信号经过长链路后太弱，模型倾向于保持均匀分配。

**可能的改进**：
1. **辅助损失**：加入分配 entropy 最小化损失或多样性正则
2. **预定义原语**：参考 TESS 论文，用领域知识预定义原语类别（趋势/波动/形态/滞后），而非纯可学习
3. **Gating 机制**：参考 TESS 的 confidence-aware gating，用置信度过滤不可靠的原语
4. **LLM 初始化**：用预训练语言模型的文本嵌入初始化 codebook，给原语语义方向的起点

### 8.2 注入方式

当前使用 cross-attention，可考虑替换为 TESS 风格的 prefix token 拼接：
- `Z = [primitive_tokens; patch_embeddings]` → 统一进 self-attention
- 优点：梯度路径更短，原语和 patch 可双向交互
- 缺点：增加了 self-attention 的序列长度

### 8.3 计算效率

逐通道循环导致推理时间为 PatchTST 的 C 倍（C=7 时约 7 倍）。可通过批处理优化：将所有通道的 patch embedding 合并为一个大 batch 一次性通过 encoder。

## 9. 文件结构

```
forecast/
├── models/
│   └── PrimitiveFusion.py        # 模型定义（本文档描述的内容）
├── layers/
│   ├── Embed.py                  # PatchEmbedding（复用）
│   └── RevIN.py                  # RevIN（复用）
├── exp/
│   ├── exp_basic.py              # 模型注册
│   └── exp_long_term_forecasting.py  # 训练/测试流程
└── run.py                        # CLI 入口
```

## 10. 使用方式

```bash
# 训练
python -m forecast.run \
  --mode single --model PrimitiveFusion --data ETTh1 \
  --seq_len 512 --pred_len 96 --train_epochs 10 --batch_size 128 \
  --d_model 128 --n_heads 8 --e_layers 3 --d_ff 256 \
  --patch_len 16 --stride 8 --num_primitives 16 --revin --use_gpu

# 仅测试（加载已保存的 checkpoint）
python -m forecast.run \
  --mode single --model PrimitiveFusion --data ETTh1 \
  --seq_len 512 --pred_len 96 --batch_size 128 \
  --d_model 128 --n_heads 8 --e_layers 3 --d_ff 256 \
  --patch_len 16 --stride 8 --num_primitives 16 --revin --use_gpu \
  --is_training 0
```

位置关系，影响关系