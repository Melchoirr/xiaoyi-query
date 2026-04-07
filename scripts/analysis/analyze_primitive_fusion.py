"""PrimitiveFusion 分析脚本：逐通道指标 + codebook 可视化 + 原语分配热力图"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from forecast.models.PrimitiveFusion import Model
from forecast.data_provider.data_factory import data_provider

# ─── 配置 ───
RESULT_DIR = 'forecast/results/PrimitiveFusion_ETTh1_M_sl512_pl96'
CKPT_PATH = 'checkpoints/PrimitiveFusion_ETTh1_M_sl512_pl96/checkpoint.pth'
CHANNEL_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
OUTPUT_DIR = 'forecast/scripts/primitive_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════
# 1. 逐通道预测指标
# ═══════════════════════════════════════════
pred = np.load(os.path.join(RESULT_DIR, 'pred.npy'))  # [N, 96, 7]
true = np.load(os.path.join(RESULT_DIR, 'true.npy'))

print('='*60)
print('逐通道预测指标 (test set)')
print('='*60)
print(f'{"Channel":<10} {"MSE":>10} {"MAE":>10}')
print('-'*30)
per_ch_mse = []
for c, name in enumerate(CHANNEL_NAMES):
    mse = np.mean((pred[:, :, c] - true[:, :, c]) ** 2)
    mae = np.mean(np.abs(pred[:, :, c] - true[:, :, c]))
    per_ch_mse.append(mse)
    print(f'{name:<10} {mse:>10.6f} {mae:>10.6f}')

overall_mse = np.mean((pred - true) ** 2)
overall_mae = np.mean(np.abs(pred - true))
print('-'*30)
print(f'{"Overall":<10} {overall_mse:>10.6f} {overall_mae:>10.6f}')

# 逐通道指标柱状图
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CHANNEL_NAMES, per_ch_mse, color='steelblue', edgecolor='white')
ax.set_ylabel('MSE')
ax.set_title('Per-Channel MSE (PrimitiveFusion, ETTh1)')
for bar, v in zip(bars, per_ch_mse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{v:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_channel_mse.png'), dpi=150)
plt.close()
print(f'\n[saved] {OUTPUT_DIR}/per_channel_mse.png')


# ═══════════════════════════════════════════
# 2. 加载模型，提取 codebook
# ═══════════════════════════════════════════
args = SimpleNamespace(
    seq_len=512, pred_len=96, enc_in=7, d_model=128, n_heads=8,
    e_layers=3, d_ff=256, patch_len=16, stride=8, dropout=0.1,
    fc_dropout=0.1, head_dropout=0.0, revin=True, num_primitives=16,
    primitive_temp=1.0, n_cross_layers=1,
)
model = Model(args)
state = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

codebook = model.primitive_assignment.codebook.detach().numpy()  # [K, d_model]
print(f'\nCodebook shape: {codebook.shape}')


# ═══════════════════════════════════════════
# 3. Codebook 可视化 (PCA 降维到 2D)
# ═══════════════════════════════════════════
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
cb_2d = pca.fit_transform(codebook)  # [K, 2]

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(cb_2d[:, 0], cb_2d[:, 1], s=120, c=range(len(cb_2d)),
                     cmap='tab20', edgecolors='black', linewidths=0.8, zorder=3)
for i, (x, y) in enumerate(cb_2d):
    ax.annotate(f'P{i}', (x, y), textcoords='offset points',
                xytext=(6, 6), fontsize=9)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Primitive Codebook (PCA 2D)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'codebook_pca.png'), dpi=150)
plt.close()
print(f'[saved] {OUTPUT_DIR}/codebook_pca.png')


# ═══════════════════════════════════════════
# 4. Codebook 相似度热力图
# ═══════════════════════════════════════════
# 归一化后计算余弦相似度
cb_norm = codebook / (np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-8)
cos_sim = cb_norm @ cb_norm.T  # [K, K]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(16))
ax.set_yticks(range(16))
ax.set_xticklabels([f'P{i}' for i in range(16)], fontsize=8)
ax.set_yticklabels([f'P{i}' for i in range(16)], fontsize=8)
ax.set_title('Primitive Codebook Cosine Similarity')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'codebook_similarity.png'), dpi=150)
plt.close()
print(f'[saved] {OUTPUT_DIR}/codebook_similarity.png')


# ═══════════════════════════════════════════
# 5. 原语分配热力图 (对 test 数据采样)
# ═══════════════════════════════════════════
# 用 test 数据跑一遍，收集每个目标通道下其他通道的 assignment weights
print('\n计算原语分配权重...')

test_args = SimpleNamespace(
    data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv',
    features='M', target='OT', freq='h',
    seq_len=512, label_len=48, pred_len=96,
    batch_size=128, num_workers=0, embed='timeF',
    src_channel=None, tgt_channel=None,
)
test_data, test_loader = data_provider(test_args, flag='test')

# 收集所有样本的 assignment weights (对每个目标通道)
# 为了效率，只取前 500 个样本
N_SAMPLES = 500
all_x = []
for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
    all_x.append(batch_x)
    if sum(b.shape[0] for b in all_x) >= N_SAMPLES:
        break
all_x = torch.cat(all_x, dim=0)[:N_SAMPLES].float()  # [N, L, C]

if model.use_revin:
    all_x = model.revin.normalize(all_x)

# 对每个目标通道，计算平均 assignment weights
avg_weights_per_target = {}  # target_ch -> [C-1, K] 平均权重

with torch.no_grad():
    for c in range(7):
        other_idx = [i for i in range(7) if i != c]
        x_o = all_x[:, :, other_idx].permute(0, 2, 1)  # [N, 6, L]
        channel_repr = model.channel_encoder(x_o)
        _, weights = model.primitive_assignment(channel_repr)  # [N, 6, K]
        avg_w = weights.mean(dim=0).numpy()  # [6, K]
        avg_weights_per_target[c] = avg_w

# 绘制: 7 个子图，每个子图是一个目标通道下 6 个其他通道的原语分配
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for c in range(7):
    ax = axes[c]
    other_names = [CHANNEL_NAMES[i] for i in range(7) if i != c]
    w = avg_weights_per_target[c]  # [6, K]
    im = ax.imshow(w, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(6))
    ax.set_yticklabels(other_names, fontsize=8)
    ax.set_xlabel('Primitive', fontsize=9)
    ax.set_title(f'Target: {CHANNEL_NAMES[c]}', fontsize=10)
    ax.set_xticks(range(0, 16, 2))

# 隐藏最后一个空子图
axes[7].axis('off')
fig.suptitle('Primitive Assignment Weights (avg over test samples)', fontsize=13, y=1.01)
plt.colorbar(im, ax=axes.tolist(), shrink=0.6, label='Weight')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'assignment_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'[saved] {OUTPUT_DIR}/assignment_heatmap.png')


# ═══════════════════════════════════════════
# 6. 原语利用率 (entropy)
# ═══════════════════════════════════════════
# 所有目标通道下的平均分配
all_weights = np.stack(list(avg_weights_per_target.values()))  # [7, 6, K]
avg_per_primitive = all_weights.mean(axis=(0, 1))  # [K]
entropy = -np.sum(avg_per_primitive * np.log(avg_per_primitive + 1e-8))
max_entropy = np.log(16)

print(f'\n原语利用率分析:')
print(f'  平均分配 entropy: {entropy:.4f} (max={max_entropy:.4f}, ratio={entropy/max_entropy:.1%})')
print(f'  各原语平均激活: {avg_per_primitive.round(4)}')

# 原语使用频率柱状图
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(16), avg_per_primitive, color='coral', edgecolor='white')
ax.axhline(y=1/16, color='gray', linestyle='--', alpha=0.7, label='Uniform (1/K)')
ax.set_xlabel('Primitive ID')
ax.set_ylabel('Avg Weight')
ax.set_title(f'Primitive Utilization (entropy={entropy:.3f}/{max_entropy:.3f})')
ax.set_xticks(range(16))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'primitive_utilization.png'), dpi=150)
plt.close()
print(f'[saved] {OUTPUT_DIR}/primitive_utilization.png')

print(f'\n所有图表已保存到 {OUTPUT_DIR}/')
