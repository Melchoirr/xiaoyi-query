"""
Streamlit 交互式可视化仪表盘
用于对比三种时序预测基线模型的性能

功能：
1. 宏观指标对比：柱状图对比不同模型在不同预测长度下的MSE/MAE
2. 微观波形探查：交互式折线图查看真实值与预测值的对比

v2.7 变更：
- RevIN 数值安全：std < 1e-5 时强制置 1.0
- 特征维度选择器：动态选择要展示的特征列（替代硬编码 -1）
- HTML flex 单行指标：display:flex 替代 st.columns()，跨屏幕强制单行
- Plotly zeroline：所有图表 Y 轴增加 zeroline=True，zerolinecolor='lightgray'
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 页面配置
st.set_page_config(
    page_title="时序预测基线模型对比",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式设置
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    /* 强制单行指标卡片：display:flex 替代 st.columns() */
    .metrics-row {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: stretch;
        gap: 0.5rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        flex: 1;
        padding: 1rem 0.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        min-width: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 0.25rem;
        white-space: nowrap;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Streamlit block 去除边距 */
    .stTabs { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 数据加载
# ============================================================

@st.cache_data(ttl=3600)
def load_experiment_log(output_dir: str = './results') -> Optional[Dict[str, Any]]:
    log_path = os.path.join(output_dir, 'experiment_log.json')
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_prediction_filename(model_name: str, seq_len: int, pred_len: int, config: dict = None) -> str:
    dataset_name = 'ETTm1'
    if model_name == 'PatternSearch':
        top_k = config.get('top_k', 5) if config else 5
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_k{top_k}"
    elif model_name == 'LSHSearch':
        n_hash = config.get('n_hash_funcs', 16) if config else 16
        n_tables = config.get('n_tables', 4) if config else 4
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_lsh_h{n_hash}_t{n_tables}"
    else:
        word_size = config.get('word_size', 8) if config else 8
        alpha = config.get('alphabet_size', 8) if config else 8
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_sax_w{word_size}_a{alpha}"


def load_predictions(
    output_dir: str, model_name: str, seq_len: int,
    pred_len: int, config: dict = None
) -> tuple:
    exp_id = _get_prediction_filename(model_name, seq_len, pred_len, config)
    preds_path = os.path.join(output_dir, f"{exp_id}_preds.npy")
    trues_path = os.path.join(output_dir, f"{exp_id}_trues.npy")
    x_test_path = os.path.join(output_dir, f"{exp_id}_X_test.npy")

    preds = np.load(preds_path) if os.path.exists(preds_path) else None
    trues = np.load(trues_path) if os.path.exists(trues_path) else None
    x_test = np.load(x_test_path) if os.path.exists(x_test_path) else None

    return preds, trues, x_test


def find_available_predictions(output_dir: str) -> List[Dict[str, Any]]:
    if not os.path.exists(output_dir):
        return []
    available = []
    for f in os.listdir(output_dir):
        if not f.endswith('_preds.npy'):
            continue
        try:
            base = f.replace('_preds.npy', '')
            parts = base.split('_')
            seq_len = pred_len = None
            model_name = None
            extra_params = {}
            for i, part in enumerate(parts):
                if part == 'seq' and i + 1 < len(parts):
                    seq_len = int(parts[i + 1])
                elif part == 'pred' and i + 1 < len(parts):
                    pred_len = int(parts[i + 1])
                elif part == 'k' and i + 1 < len(parts):
                    model_name = 'PatternSearch'
                    extra_params['top_k'] = int(parts[i + 1])
                elif part == 'lsh':
                    model_name = 'LSHSearch'
                    extra_params['n_hash_funcs'] = int(parts[i + 1][1:])
                    extra_params['n_tables'] = int(parts[i + 2][1:])
                elif part == 'sax':
                    model_name = 'SAXSearch'
                    extra_params['word_size'] = int(parts[i + 1][1:])
                    extra_params['alphabet_size'] = int(parts[i + 2][1:])
            if model_name and seq_len and pred_len:
                available.append({
                    'model_name': model_name, 'seq_len': seq_len,
                    'pred_len': pred_len, 'exp_id': base, **extra_params
                })
        except Exception:
            continue
    return available


# ============================================================
# 页面组件
# ============================================================

def render_header():
    st.markdown('<p class="main-title">📈 时序预测基线模型对比仪表盘</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">PatternSearch vs LSHSearch vs SAXSearch | 消融实验可视化分析</p>',
        unsafe_allow_html=True
    )


def render_sidebar() -> Dict[str, Any]:
    """
    侧边栏（含 expander 表单 + 动态 Sample ID 范围）

    任务四：表单折叠在 st.sidebar.expander("🚀 启动新实验", expanded=False)
    """
    st.sidebar.markdown("## ⚙️ 配置选项")

    output_dir = st.sidebar.text_input(
        "结果目录", value="./results",
        help="实验结果 JSON 文件所在目录"
    )

    # ─────────────────────────────────────────────────────────────────
    # 表单：封装在 expander 中，默认折叠
    # ─────────────────────────────────────────────────────────────────
    with st.sidebar.expander("🚀 启动新实验", expanded=False):
        with st.form("experiment_form", clear_on_submit=False):
            st.markdown("**快速启动配置**")

            run_model = st.selectbox(
                "模型",
                options=['PatternSearch', 'LSHSearch', 'SAXSearch', 'all'],
                index=3, help="选择要运行的模型"
            )

            # PatternSearch 参数
            if run_model in ('PatternSearch', 'all'):
                with st.expander("PatternSearch 参数", expanded=False):
                    run_top_k = st.number_input(
                        "top_k（近邻数）", min_value=1, max_value=100,
                        value=5, step=1
                    )
                    run_weighted_ps = st.checkbox("weighted（逆距离加权）", value=True)

            # LSHSearch 参数
            if run_model in ('LSHSearch', 'all'):
                with st.expander("LSHSearch 参数", expanded=False):
                    run_n_hash = st.number_input(
                        "n_hash_funcs（哈希函数数）", min_value=1, max_value=64,
                        value=16, step=1
                    )
                    run_n_tables = st.number_input(
                        "n_tables（哈希表数量）", min_value=1, max_value=16,
                        value=4, step=1
                    )
                    run_lsh_cap = st.number_input(
                        "candidate_cap_total（候选上限）", min_value=64, max_value=4096,
                        value=1024, step=64
                    )
                    run_lsh_weighted = st.checkbox("lsh_weighted（加权重排）", value=False)

            # SAXSearch 参数
            if run_model in ('SAXSearch', 'all'):
                with st.expander("SAXSearch 参数", expanded=False):
                    run_word_size = st.number_input(
                        "word_size（词大小）", min_value=2, max_value=32,
                        value=8, step=1
                    )
                    run_alpha_size = st.number_input(
                        "alphabet_size（字母表大小）", min_value=2, max_value=32,
                        value=8, step=1
                    )
                    run_bucket_k = st.number_input(
                        "bucket_top_k（桶内近邻数）", min_value=1, max_value=32,
                        value=8, step=1
                    )
                    run_sax_weighted = st.checkbox("sax_weighted（加权聚合）", value=True)

            # 全局参数
            run_seq_len = st.number_input(
                "seq_len（输入长度）", min_value=1, max_value=10000,
                value=96, step=1,
                help="TSLib 常用值：96, 192, 336, 720（支持任意正整数）"
            )
            run_pred_len = st.number_input(
                "pred_len（预测长度）", min_value=1, max_value=10000,
                value=48, step=1,
                help="TSLib 常用值：96, 192, 336, 720（支持任意正整数）"
            )
            run_revin = st.checkbox(
                "启用 RevIN（可逆实例归一化）", value=False,
                help="训练/推理执行 (X-mean)/std 归一化，预测后 Y_pred*std+mean 反归一化"
            )
            run_gpu = st.checkbox("启用 GPU 加速", value=False)

            submitted = st.form_submit_button(
                "▶️ 运行实验",
                type="primary",
                use_container_width=True
            )

            # ── 核心：subprocess.Popen 非阻塞启动 ──
            if submitted:
                import subprocess
                import sys as _sys

                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                run_py = os.path.join(project_root, 'run.py')

                cmd = [
                    _sys.executable, run_py,
                    "--model", run_model,
                    "--seq_len", str(int(run_seq_len)),
                    "--pred_len", str(int(run_pred_len)),
                ]
                if run_gpu:
                    cmd.append("--use_gpu")
                if run_revin:
                    cmd.append("--revin")

                if run_model == 'PatternSearch':
                    cmd.extend(["--top_k", str(int(run_top_k))])
                    cmd.extend(["--weighted", str(run_weighted_ps).lower()])
                elif run_model == 'LSHSearch':
                    cmd.extend(["--n_hash_funcs", str(int(run_n_hash))])
                    cmd.extend(["--n_tables", str(int(run_n_tables))])
                    cmd.extend(["--candidate_cap_total", str(int(run_lsh_cap))])
                    cmd.extend(["--lsh_weighted", str(run_lsh_weighted).lower()])
                elif run_model == 'SAXSearch':
                    cmd.extend(["--word_size", str(int(run_word_size))])
                    cmd.extend(["--alphabet_size", str(int(run_alpha_size))])
                    cmd.extend(["--bucket_top_k", str(int(run_bucket_k))])
                    cmd.extend(["--sax_weighted", str(run_sax_weighted).lower()])

                st.info(f"执行命令：`{' '.join(cmd)}`")

                log_placeholder = st.empty()
                log_lines: list = []

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=project_root
                    )

                    spinner_ph = st.empty()
                    for raw_line in iter(process.stdout.readline, ''):
                        if not raw_line:
                            break
                        line = raw_line.rstrip()
                        log_lines.append(line)
                        if len(log_lines) > 200:
                            log_lines = log_lines[-200:]
                        spinner_ph.info("⏳ 实验运行中（见右侧日志）...")
                        with log_placeholder.container():
                            st.code("\n".join(log_lines[-80:]), language=None, height=320)

                    process.wait()

                    with log_placeholder.container():
                        if process.returncode == 0:
                            spinner_ph.success("✅ 实验完成！")
                            st.success("✅ 实验完成！正在刷新页面...")
                            st.rerun()
                        else:
                            spinner_ph.error(f"❌ 失败 (exit {process.returncode})")
                            st.error(f"❌ 实验失败（exit {process.returncode}）")
                            st.code("\n".join(log_lines[-50:]), language=None, height=200)
                except Exception as ex:
                    st.error(f"❌ 启动失败: {ex}")

    # ── 已有实验结果的可视化选择器 ───────────────────────────────
    log_data = load_experiment_log(output_dir)

    if log_data is None:
        st.sidebar.warning("⚠️ 未找到实验日志文件，请先运行实验")
        return {
            'output_dir': output_dir, 'log_data': None,
            'selected_model': None, 'selected_pred_len': None,
            'selected_seq_len': None, 'selected_sample_id': 0,
            'experiments': [], 'max_preview_count': 100,
            'n_features': 1, 'selected_feature_idx': 0
        }

    st.sidebar.success("✅ 实验日志已加载")

    metadata = log_data.get('metadata', {})
    with st.sidebar.expander("📊 实验元数据", expanded=False):
        st.write(f"**数据集:** {metadata.get('dataset', 'N/A')}")
        st.write(f"**特征模式:** {metadata.get('features', 'N/A')}")
        st.write(f"**目标列:** {metadata.get('target', 'N/A')}")
        st.write(f"**总实验数:** {metadata.get('total', 0)}")
        st.write(f"**时间戳:** {metadata.get('timestamp', 'N/A')[:19]}")

    experiments = log_data.get('experiments', [])
    available_models = sorted(list(set(
        exp['config']['model_name'] for exp in experiments
        if exp['status'] == 'success'
    )))

    if not available_models:
        st.sidebar.warning("⚠️ 没有成功的实验")
        return {
            'output_dir': output_dir, 'log_data': log_data,
            'selected_model': None, 'selected_pred_len': None,
            'selected_seq_len': None, 'selected_sample_id': 0,
            'experiments': experiments, 'max_preview_count': 100,
            'n_features': 1, 'selected_feature_idx': 0
        }

    selected_model = st.sidebar.selectbox(
        "选择模型", options=available_models, index=0
    )

    pred_lens = sorted(list(set(
        exp['config']['pred_len']
        for exp in experiments
        if exp['config']['model_name'] == selected_model and exp['status'] == 'success'
    )))
    selected_pred_len = st.sidebar.selectbox(
        "预测长度 (pred_len)", options=pred_lens, index=0
    )

    seq_lens = sorted(list(set(
        exp['config']['seq_len']
        for exp in experiments
        if exp['config']['model_name'] == selected_model
        and exp['config']['pred_len'] == selected_pred_len
        and exp['status'] == 'success'
    )))
    selected_seq_len = st.sidebar.selectbox(
        "序列长度 (seq_len)", options=seq_lens, index=0
    )

    # 推断 n_features
    n_features = 1
    for exp in experiments:
        if (exp['status'] == 'success'
            and exp['config']['model_name'] == selected_model
            and exp['config']['pred_len'] == selected_pred_len
            and exp['config']['seq_len'] == selected_seq_len
            and 'preview' in exp):
            preview = exp['preview']
            if preview and 'history' in preview and len(preview.get('history', [[]])) > 0:
                hist_len = len(preview['history'][0])
                n_features = hist_len // selected_seq_len \
                    if hist_len % selected_seq_len == 0 else 1
            break

    selected_feature_idx = st.sidebar.selectbox(
        "选择展示的特征维度",
        options=list(range(max(1, n_features))),
        index=max(0, n_features - 1),
        format_func=lambda x: f"特征列 #{x} (Target)" if x == n_features - 1 else f"特征列 #{x}",
        help="选择要绘图的特征维度（默认最后一列为 Target/OT）"
    )

    # Sample ID 范围：动态从 preview['count'] 读取
    max_preview_count = 100
    for exp in experiments:
        if (exp['status'] == 'success'
            and exp['config']['model_name'] == selected_model
            and exp['config']['pred_len'] == selected_pred_len
            and exp['config']['seq_len'] == selected_seq_len
            and 'preview' in exp):
            max_preview_count = max(
                max_preview_count, exp['preview'].get('count', 100)
            )
            break

    selected_sample_id = st.sidebar.slider(
        "样本 ID",
        min_value=0,
        max_value=max(1, max_preview_count - 1),
        value=0,
        help=f"选择要查看的样本索引（0-{max_preview_count - 1}）"
    )

    return {
        'output_dir': output_dir,
        'log_data': log_data,
        'selected_model': selected_model,
        'selected_pred_len': selected_pred_len,
        'selected_seq_len': selected_seq_len,
        'selected_sample_id': selected_sample_id,
        'selected_feature_idx': selected_feature_idx,
        'n_features': n_features,
        'experiments': experiments,
        'max_preview_count': max_preview_count
    }


# ============================================================
# HTML 指标渲染工具
# ============================================================

METRIC_CARDS_HTML = """
<div class="metrics-row">
    <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{mae:.4f}</div>
    </div>
    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="metric-label">MSE</div>
        <div class="metric-value">{mse:.4f}</div>
    </div>
    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{rmse:.4f}</div>
    </div>
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-label">MAPE</div>
        <div class="metric-value">{mape:.4f}</div>
    </div>
    <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="metric-label">CORR</div>
        <div class="metric-value">{corr:.4f}</div>
    </div>
</div>
"""


def render_metrics_html(mae: float, mse: float, rmse: float, mape: float, corr: float):
    """用 display:flex HTML 卡片渲染 5 个指标，强制单行排布"""
    st.markdown(
        METRIC_CARDS_HTML.format(
            mae=mae, mse=mse, rmse=rmse, mape=mape, corr=corr
        ),
        unsafe_allow_html=True
    )


# ============================================================
# 宏观指标对比
# ============================================================

def render_metrics_comparison(log_data: Dict[str, Any]):
    """
    渲染宏观指标对比

    任务三：指标用 HTML flex 替代 st.columns()，强制单行
    任务四：所有图表增加 zeroline
    """
    st.markdown("---")
    st.markdown("## 📊 宏观指标对比")

    experiments = log_data.get('experiments', [])
    success_exps = [e for e in experiments if e['status'] == 'success']
    if not success_exps:
        st.warning("没有成功的实验结果")
        return

    df_data = []
    for exp in success_exps:
        df_data.append({
            'Model': exp['config']['model_name'],
            'seq_len': exp['config']['seq_len'],
            'pred_len': exp['config']['pred_len'],
            'MAE': exp['metrics'].get('MAE', 0),
            'MSE': exp['metrics'].get('MSE', 0),
            'RMSE': exp['metrics'].get('RMSE', 0),
            'MAPE': exp['metrics'].get('MAPE', 0),
        })

    df = pd.DataFrame(df_data)
    agg_df = df.groupby(['Model', 'pred_len']).agg({
        'MAE': 'mean', 'MSE': 'mean', 'RMSE': 'mean', 'MAPE': 'mean'
    }).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 MAE 对比")
        fig_mae = px.bar(
            agg_df, x='pred_len', y='MAE', color='Model',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                'pred_len': '预测长度',
                'MAE': 'MAE (越低越好)',
                'Model': '模型'
            },
            title='不同预测长度下的 MAE 对比'
        )
        fig_mae.update_layout(
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            xaxis=dict(tickmode='array', tickvals=[24, 48, 96]),
            height=400,
            yaxis=dict(zeroline=True, zerolinecolor='lightgray')
        )
        fig_mae.update_traces(texttemplate='%{y:.4f}', textposition='outside')
        st.plotly_chart(fig_mae, width="stretch")

    with col2:
        st.markdown("### 📉 MSE 对比")
        fig_mse = px.bar(
            agg_df, x='pred_len', y='MSE', color='Model',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                'pred_len': '预测长度',
                'MSE': 'MSE (越低越好)',
                'Model': '模型'
            },
            title='不同预测长度下的 MSE 对比'
        )
        fig_mse.update_layout(
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            xaxis=dict(tickmode='array', tickvals=[24, 48, 96]),
            height=400,
            yaxis=dict(zeroline=True, zerolinecolor='lightgray')
        )
        fig_mse.update_traces(texttemplate='%{y:.4f}', textposition='outside')
        st.plotly_chart(fig_mse, width="stretch")

    st.markdown("### 📈 MAPE 对比")
    fig_mape = px.bar(
        agg_df, x='pred_len', y='MAPE', color='Model',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={
            'pred_len': '预测长度',
            'MAPE': 'MAPE (%) (越低越好)',
            'Model': '模型'
        },
        title='不同预测长度下的 MAPE 对比'
    )
    fig_mape.update_layout(
        template='plotly_white',
        xaxis=dict(tickmode='array', tickvals=[24, 48, 96]),
        height=350,
        yaxis=dict(zeroline=True, zerolinecolor='lightgray')
    )
    fig_mape.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    st.plotly_chart(fig_mape, width="stretch")


# ============================================================
# 微观波形对比
# ============================================================

def _infer_n_features(preview: dict, seq_len: int) -> int:
    """从 preview['history'] 推断 n_features"""
    if preview and 'history' in preview and len(preview.get('history', [[]])) > 0:
        hist_len = len(preview['history'][0])
        return hist_len // seq_len if hist_len % seq_len == 0 else 1
    return 1


def _extract_feature(data: np.ndarray, feat_idx: int) -> np.ndarray:
    """
    从 2D/3D 数据中提取指定特征列（索引 feat_idx）

    Args:
        data: shape (n, seq_or_pred) 或 (n, seq_or_pred, n_feat)
        feat_idx: 要提取的特征列索引

    Returns:
        shape (n, seq_or_pred) 的 2D 数组
    """
    if data.ndim == 3:
        return data[:, :, feat_idx]
    elif data.ndim == 2:
        return data[:, feat_idx:feat_idx + 1] if feat_idx < data.shape[1] else data
    else:
        return data.flatten()


def _restore_feature(raw: np.ndarray, n_features: int, seq_len: int, feat_idx: int) -> np.ndarray:
    """
    从 JSON preview 展平数据中提取指定特征列

    preview['history'] = X_test[:100].reshape(100, -1)  # 展平为 (100, seq_len * n_feat)

    Returns:
        shape (seq_len,) 的 1D 数组
    """
    if n_features <= 1 or feat_idx >= n_features:
        return raw.flatten()

    total_len = len(raw)
    per_feat = total_len // n_features
    start = feat_idx * per_feat
    end = start + per_feat
    return raw[start:end]


def _base_figure_layout(
    fig: go.Figure,
    title: str,
    seq_len: int,
    pred_len_actual: int,
    mae: float = 0.0,
    mse: float = 0.0,
) -> go.Figure:
    """统一应用：zeroline、xaxis 范围、历史/未来分隔线"""
    total_x_range = seq_len + pred_len_actual
    fig.update_layout(
        title=title,
        xaxis_title='时间步（0 = 预测起点 | 负值 = 历史）',
        yaxis_title='值',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            tickmode='linear', tick0=-seq_len,
            dtick=max(1, total_x_range // 12)
        ),
        # 任务四：Y 轴 zeroline，穿越 0 轴时清晰可见
        yaxis=dict(zeroline=True, zerolinecolor='lightgray', zerolinewidth=1.5)
    )
    return fig


def render_waveform_comparison(
    config: Dict[str, Any],
    experiments: Optional[List[Dict[str, Any]]] = None
):
    """
    渲染微观波形对比

    任务二：特征维度选择器（动态下拉框，替代硬编码 -1）
    任务三：指标 HTML flex 强制单行（替代 st.columns(5)）
    任务四：所有图表 zeroline，x=0 分隔虚线
    """
    st.markdown("---")
    st.markdown("## 🔍 微观波形探查 (Case Study)")

    if experiments is None:
        experiments = config.get('experiments')
    if experiments is None and config.get('log_data') is not None:
        experiments = config['log_data'].get('experiments', [])
    if experiments is None:
        experiments = []

    model = config.get('selected_model')
    pred_len = config.get('selected_pred_len')
    seq_len = config.get('selected_seq_len')
    sample_id = config.get('selected_sample_id', 0)
    output_dir = config.get('output_dir', './results')
    feat_idx = config.get('selected_feature_idx', 0)
    n_features_total = config.get('n_features', 1)

    if model is None or pred_len is None or seq_len is None:
        st.info("💡 请从侧边栏选择模型和参数后查看波形。")
        return

    is_multi_feat = n_features_total > 1

    st.markdown(f"""
    <div class="info-panel">
        <strong>当前配置:</strong> {model} | seq_len={seq_len} | pred_len={pred_len}
        | 样本 #{sample_id} | 特征列 #{feat_idx}{" (Target)" if feat_idx == n_features_total - 1 and is_multi_feat else ""}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── 匹配实验，读取 metrics ───────────────────────────────────
    matching_exp = None
    for exp in experiments:
        if (exp['status'] == 'success'
            and exp['config']['model_name'] == model
            and exp['config']['pred_len'] == pred_len
            and exp['config']['seq_len'] == seq_len):
            matching_exp = exp
            break

    if matching_exp is not None:
        m = matching_exp.get('metrics', {})
        mae = m.get('MAE', 0.0)
        mse = m.get('MSE', 0.0)
        rmse = m.get('RMSE', 0.0)
        mape = m.get('MAPE', 0.0)
        corr = m.get('CORR', 0.0)
    else:
        mae = mse = rmse = mape = corr = 0.0

    # ── 任务三（核心）：HTML flex 指标卡片，强制单行 ─────────────
    render_metrics_html(mae, mse, rmse, mape, corr)
    st.markdown("")

    # ── 多特征说明注释 ─────────────────────────────────────────
    if is_multi_feat:
        st.info(
            f"📌 **多变量 (M) 模式**：数据包含 {n_features_total} 个特征变量。"
            f"当前展示 **特征列 #{feat_idx}**"
            f"{'（Target / OT 列）' if feat_idx == n_features_total - 1 else ''}。"
            "可通过侧边栏「选择展示的特征维度」切换其他特征。"
        )

    # ── 尝试从 JSON preview 读取连贯波形数据 ───────────────────
    preview = None
    if matching_exp and 'preview' in matching_exp:
        preview = matching_exp['preview']

    if preview is not None and 'history' in preview and 'trues' in preview and 'preds' in preview:
        history_list = preview['history']
        true_list = preview['trues']
        pred_list = preview['preds']

        if (sample_id < len(history_list)
            and sample_id < len(true_list)
            and sample_id < len(pred_list)):

            # 任务二：从展平数据中提取 selected_feature_idx 列
            hist_sample = _restore_feature(
                np.array(history_list[sample_id]),
                n_features_total, seq_len, feat_idx
            )
            true_sample = _restore_feature(
                np.array(true_list[sample_id]),
                n_features_total, pred_len, feat_idx
            )
            pred_sample = _restore_feature(
                np.array(pred_list[sample_id]),
                n_features_total, pred_len, feat_idx
            )

            # 连贯波形
            st.markdown(f"### 📈 连贯波形（样本 #{sample_id} | 特征 #{feat_idx}）")

            fig = go.Figure()

            # 历史（[-seq_len, -1]，gray）
            hist_x = list(range(-seq_len, 0))
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_sample.tolist(), mode='lines',
                name='历史真实 (History)',
                line=dict(color='#B0B0B0', width=2.0), opacity=0.8,
                hovertemplate='时间: %{x}<br>历史值: %{y:.4f}<extra></extra>'
            ))

            # 真实未来（[0, pred_len-1]，blue）
            future_x = list(range(0, len(true_sample)))
            fig.add_trace(go.Scatter(
                x=future_x, y=true_sample.tolist(), mode='lines+markers',
                name='未来真实 (Ground Truth)',
                line=dict(color='#1F4E79', width=3.0),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='时间: %{x}<br>真实值: %{y:.4f}<extra></extra>'
            ))

            # 预测（[0, pred_len-1]，red 虚线）
            fig.add_trace(go.Scatter(
                x=future_x, y=pred_sample.tolist(), mode='lines+markers',
                name='模型预测 (Prediction)',
                line=dict(color='#C00000', width=3.0, dash='dash'),
                marker=dict(size=7, symbol='square'),
                hovertemplate='时间: %{x}<br>预测值: %{y:.4f}<extra></extra>'
            ))

            # x=0 分隔虚线
            fig.add_vline(x=0, line_dash="dot", line_color="#404040", line_width=2.0)

            # 误差区域
            fig.add_trace(go.Scatter(
                x=future_x + future_x[::-1],
                y=(pred_sample - true_sample).tolist() + [0] * len(future_x),
                fill='toself', fillcolor='rgba(220, 80, 80, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='预测误差带', showlegend=True,
                hovertemplate='时间: %{x}<br>误差: %{y:.4f}<extra></extra>'
            ))

            _base_figure_layout(
                fig,
                title=f'{model} | seq={seq_len} | pred={len(true_sample)} | '
                      f'特征#{feat_idx} | MAE={mae:.4f} | MSE={mse:.4f}',
                seq_len=seq_len,
                pred_len_actual=len(true_sample),
                mae=mae, mse=mse
            )
            st.plotly_chart(fig, width="stretch")

            # 数值统计
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("历史均值", f"{float(np.mean(hist_sample)):.4f}")
            with s2:
                st.metric("未来均值", f"{float(np.mean(true_sample)):.4f}")
            with s3:
                st.metric("预测均值", f"{float(np.mean(pred_sample)):.4f}")
            with s4:
                st.metric("样本 ID", f"#{sample_id}")

            # 多样本批量对比
            st.markdown("### 📊 多样本批量对比视图")
            n_multi = 5
            start_idx = max(0, sample_id - 2)

            fig_multi = go.Figure()
            colors_true = px.colors.qualitative.Set1[:n_multi]
            colors_pred = px.colors.qualitative.Dark2[:n_multi]

            for i in range(n_multi):
                idx = start_idx + i
                if (idx < len(history_list)
                    and idx < len(true_list)
                    and idx < len(pred_list)):
                    t_s = _restore_feature(
                        np.array(true_list[idx]), n_features_total, pred_len, feat_idx
                    )
                    p_s = _restore_feature(
                        np.array(pred_list[idx]), n_features_total, pred_len, feat_idx
                    )

                    offset = i * (len(t_s) + 5)
                    fig_multi.add_trace(go.Scatter(
                        x=[x + offset for x in range(len(t_s))], y=t_s.tolist(),
                        mode='lines', name=f'真实 #{idx}',
                        line=dict(color=colors_true[i], width=2), showlegend=True
                    ))
                    fig_multi.add_trace(go.Scatter(
                        x=[x + offset for x in range(len(p_s))], y=p_s.tolist(),
                        mode='lines', name=f'预测 #{idx}',
                        line=dict(color=colors_pred[i], width=2, dash='dash'), showlegend=True
                    ))

            fig_multi.update_layout(
                title=f'连续 5 个样本的预测对比（特征 #{feat_idx}）',
                xaxis_title='时间步（带偏移）',
                yaxis_title='值',
                template='plotly_white',
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.15, xanchor='right', x=1),
                yaxis=dict(zeroline=True, zerolinecolor='lightgray', zerolinewidth=1.5)
            )
            st.plotly_chart(fig_multi, width="stretch")

        else:
            st.error(f"样本 ID {sample_id} 超出范围")

    else:
        # 无 JSON preview：从 .npy 加载
        config_params = matching_exp['config'] if matching_exp else None
        preds, trues, x_test = load_predictions(
            output_dir, model, seq_len, pred_len, config_params
        )

        if preds is None or trues is None:
            st.info("💡 提示: 预测结果文件不存在，请确保已运行相应实验。")
            st.markdown("### 📋 示例波形展示（模拟数据）")

            n_timesteps = pred_len
            true_wave = np.sin(np.linspace(0, 4 * np.pi, n_timesteps)) * 2 \
                + np.random.randn(n_timesteps) * 0.3
            if model == 'PatternSearch':
                pred_wave = true_wave + np.random.randn(n_timesteps) * 0.2
            elif model == 'LSHSearch':
                pred_wave = true_wave + np.random.randn(n_timesteps) * 0.4
            else:
                pred_wave = true_wave + np.random.randn(n_timesteps) * 0.5

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(n_timesteps)), y=true_wave.tolist(),
                mode='lines+markers', name='真实值 (Ground Truth)',
                line=dict(color='#1F4E79', width=2.5), marker=dict(size=6),
                hovertemplate='时间点: %{x}<br>真实值: %{y:.4f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(n_timesteps)), y=pred_wave.tolist(),
                mode='lines+markers', name='预测值 (Prediction)',
                line=dict(color='#C00000', width=2.5, dash='dash'), marker=dict(size=6),
                hovertemplate='时间点: %{x}<br>预测值: %{y:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=f'{model} 模型预测效果（示例数据）',
                xaxis_title='时间步 (Timestep)',
                yaxis_title='值 (Value)',
                template='plotly_white',
                hovermode='x unified',
                height=450,
                margin=dict(l=40, r=40, t=60, b=40),
                yaxis=dict(zeroline=True, zerolinecolor='lightgray', zerolinewidth=1.5)
            )
            st.plotly_chart(fig, width="stretch")
        else:
            # 从 .npy 加载：用 selected_feature_idx 动态切片
            if sample_id < len(preds) and sample_id < len(trues):
                pred_s = _extract_feature(preds[sample_id], feat_idx)
                true_s = _extract_feature(trues[sample_id], feat_idx)

                hist_s = None
                if x_test is not None and sample_id < len(x_test):
                    h = _extract_feature(x_test[sample_id], feat_idx)
                    hist_s = h.flatten() if h.ndim > 1 else h

                st.markdown(
                    f"### 📈 预测结果波形（样本 #{sample_id} | 特征 #{feat_idx}）"
                )
                fig = go.Figure()

                if hist_s is not None and len(hist_s) == seq_len:
                    fig.add_trace(go.Scatter(
                        x=list(range(-seq_len, 0)), y=hist_s.tolist(), mode='lines',
                        name='历史真实 (History)',
                        line=dict(color='#B0B0B0', width=2.0), opacity=0.8,
                        hovertemplate='时间: %{x}<br>历史值: %{y:.4f}<extra></extra>'
                    ))

                fx = list(range(0, len(true_s)))
                fig.add_trace(go.Scatter(
                    x=fx, y=true_s.flatten().tolist() if true_s.ndim > 1 else true_s.tolist(),
                    mode='lines+markers', name='未来真实 (Ground Truth)',
                    line=dict(color='#1F4E79', width=3.0),
                    marker=dict(size=7, symbol='circle'),
                    hovertemplate='时间: %{x}<br>真实值: %{y:.4f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=fx, y=pred_s.flatten().tolist() if pred_s.ndim > 1 else pred_s.tolist(),
                    mode='lines+markers', name='模型预测 (Prediction)',
                    line=dict(color='#C00000', width=3.0, dash='dash'),
                    marker=dict(size=7, symbol='square'),
                    hovertemplate='时间: %{x}<br>预测值: %{y:.4f}<extra></extra>'
                ))
                fig.add_vline(x=0, line_dash="dot", line_color="#404040", line_width=2.0)

                mae_v = float(np.mean(np.abs(
                    pred_s.flatten() - true_s.flatten()
                )))
                mse_v = float(np.mean((
                    pred_s.flatten() - true_s.flatten()
                ) ** 2))

                fig.add_trace(go.Scatter(
                    x=fx + fx[::-1],
                    y=(pred_s.flatten() - true_s.flatten()).tolist() + [0] * len(fx),
                    fill='toself', fillcolor='rgba(220, 80, 80, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='预测误差带', showlegend=True,
                    hovertemplate='时间: %{x}<br>误差: %{y:.4f}<extra></extra>'
                ))

                _base_figure_layout(
                    fig,
                    title=f'{model} | seq={seq_len} | pred={len(true_s.flatten())} | '
                          f'特征#{feat_idx} | MAE={mae_v:.4f} | MSE={mse_v:.4f}',
                    seq_len=seq_len,
                    pred_len_actual=len(true_s.flatten()),
                    mae=mae_v, mse=mse_v
                )
                st.plotly_chart(fig, width="stretch")

                ss1, ss2, ss3, ss4 = st.columns(4)
                with ss1:
                    st.metric("真实均值", f"{float(np.mean(true_s.flatten())):.4f}")
                with ss2:
                    st.metric("预测均值", f"{float(np.mean(pred_s.flatten())):.4f}")
                with ss3:
                    st.metric("MAE", f"{mae_v:.4f}")
                with ss4:
                    st.metric("MSE", f"{mse_v:.4f}")
            else:
                st.error(f"样本 ID {sample_id} 超出范围")


# ============================================================
# 模型介绍
# ============================================================

def render_model_introduction():
    st.markdown("---")
    st.markdown("## 📚 模型介绍")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### PatternSearch
        **基于 torch.cdist KNN 检索**
        - 欧氏距离，逆距离加权
        - GPU 加速，torch.topk
        """)
    with col2:
        st.markdown("""
        ### LSHSearch
        **局部敏感哈希检索**
        - 随机投影，uint64 打包
        - Hamming 半径探针
        """)
    with col3:
        st.markdown("""
        ### SAXSearch
        **符号聚合近似检索**
        - PAA 降维，高斯分位数字符化
        - NearestNeighbors 模糊匹配
        """)


# ============================================================
# 主函数
# ============================================================

def main():
    render_header()
    config = render_sidebar()
    log_data = config.get('log_data')

    if log_data is None:
        st.warning("⚠️ 请先运行实验生成结果后查看")
        st.markdown("---")
        render_model_introduction()
        return

    render_metrics_comparison(log_data)
    render_waveform_comparison(
        config,
        experiments=log_data.get('experiments', [])
    )
    render_model_introduction()


if __name__ == '__main__':
    main()
