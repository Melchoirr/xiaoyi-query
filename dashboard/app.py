"""
Streamlit 交互式可视化仪表盘
用于对比三种时序预测基线模型的性能

功能：
1. 宏观指标对比：柱状图对比不同模型在不同预测长度下的MSE/MAE
2. 微观波形探查：交互式折线图查看真实值与预测值的对比

使用方法：
    streamlit run dashboard/app.py
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
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* 副标题样式 */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }

    /* 成功指标样式 */
    .metric-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    /* 警告指标样式 */
    .metric-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    /* 信息面板样式 */
    .info-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }

    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 数据加载
# ============================================================

@st.cache_data(ttl=3600)
def load_experiment_log(output_dir: str = './results') -> Optional[Dict[str, Any]]:
    """
    加载实验日志

    Args:
        output_dir: 结果目录

    Returns:
        实验日志字典，如果文件不存在返回None
    """
    log_path = os.path.join(output_dir, 'experiment_log.json')

    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_prediction_filename(model_name: str, seq_len: int, pred_len: int, config: dict = None) -> str:
    """
    根据模型类型生成预测文件名

    Args:
        model_name: 模型名称
        seq_len: 序列长度
        pred_len: 预测长度
        config: 可选的配置字典

    Returns:
        文件名前缀（不含扩展名）
    """
    dataset_name = 'ETTm1'  # 默认数据集名

    if model_name == 'PatternSearch':
        top_k = config.get('top_k', 5) if config else 5
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_k{top_k}"
    elif model_name == 'LSHSearch':
        n_hash = config.get('n_hash_funcs', 16) if config else 16
        n_tables = config.get('n_tables', 4) if config else 4
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_lsh_h{n_hash}_t{n_tables}"
    else:  # SAXSearch
        word_size = config.get('word_size', 8) if config else 8
        alpha = config.get('alphabet_size', 8) if config else 8
        return f"{dataset_name}_seq{seq_len}_pred{pred_len}_sax_w{word_size}_a{alpha}"


def load_predictions(
    output_dir: str,
    model_name: str,
    seq_len: int,
    pred_len: int,
    config: dict = None
) -> tuple:
    """
    加载预测结果、真实值、以及历史输入序列（任务7：历史上下文）

    Returns:
        (preds, trues, x_test) 元组
    """
    # 生成文件名
    exp_id = _get_prediction_filename(model_name, seq_len, pred_len, config)

    preds_path = os.path.join(output_dir, f"{exp_id}_preds.npy")
    trues_path = os.path.join(output_dir, f"{exp_id}_trues.npy")
    x_test_path = os.path.join(output_dir, f"{exp_id}_X_test.npy")

    preds = None
    trues = None
    x_test = None

    if os.path.exists(preds_path):
        preds = np.load(preds_path)

    if os.path.exists(trues_path):
        trues = np.load(trues_path)

    if os.path.exists(x_test_path):
        x_test = np.load(x_test_path)

    return preds, trues, x_test


def find_available_predictions(output_dir: str) -> List[Dict[str, Any]]:
    """
    扫描结果目录，找到所有可用的预测文件

    Args:
        output_dir: 结果目录

    Returns:
        可用预测文件的配置列表
    """
    if not os.path.exists(output_dir):
        return []

    available = []
    files = os.listdir(output_dir)

    for f in files:
        if not f.endswith('_preds.npy'):
            continue

        # 解析文件名
        # 格式: ETTm1_seq{seq_len}_pred{pred_len}_k{top_k}_preds.npy
        # 或: ETTm1_seq{seq_len}_pred{pred_len}_lsh_h{hash}_t{tables}_preds.npy
        # 或: ETTm1_seq{seq_len}_pred{pred_len}_sax_w{word}_a{alpha}_preds.npy

        try:
            # 去掉后缀
            base = f.replace('_preds.npy', '')

            # 提取基本信息
            parts = base.split('_')
            dataset = parts[0]

            # 找到 seq 和 pred
            seq_len = None
            pred_len = None
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
                    # lsh_h{n_hash}_t{n_tables}
                    h_part = parts[i + 1]  # h{hash}
                    t_part = parts[i + 2]  # t{tables}
                    extra_params['n_hash_funcs'] = int(h_part[1:])
                    extra_params['n_tables'] = int(t_part[1:])
                elif part == 'sax':
                    model_name = 'SAXSearch'
                    # sax_w{word}_a{alpha}
                    w_part = parts[i + 1]  # w{word}
                    a_part = parts[i + 2]  # a{alpha}
                    extra_params['word_size'] = int(w_part[1:])
                    extra_params['alphabet_size'] = int(a_part[1:])

            if model_name and seq_len and pred_len:
                available.append({
                    'model_name': model_name,
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'exp_id': base,
                    **extra_params
                })
        except Exception:
            continue

    return available


# ============================================================
# 页面组件
# ============================================================

def render_header():
    """渲染页面头部"""
    st.markdown('<p class="main-title">📈 时序预测基线模型对比仪表盘</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">PatternSearch vs LSHSearch vs SAXSearch | 消融实验可视化分析</p>',
        unsafe_allow_html=True
    )


def render_sidebar() -> Dict[str, Any]:
    """
    渲染侧边栏（含 st.form 一键启动 + 实时终端 Log）

    任务三：st.form + subprocess.Popen 实时打屏，不阻塞 UI
    任务四（表单部分）：允许手填 seq_len / pred_len（覆盖预设列表）
    """
    st.sidebar.markdown("## ⚙️ 配置选项")

    # 输出目录
    output_dir = st.sidebar.text_input(
        "结果目录",
        value="./results",
        help="实验结果 JSON 文件所在目录"
    )

    # ─────────────────────────────────────────────────────────────────
    # 任务三：st.form 一键启动 + 实时 Log（非阻塞 Popen）
    # ─────────────────────────────────────────────────────────────────
    with st.sidebar.form("experiment_form", clear_on_submit=False):
        st.markdown("### 🚀 一键启动实验")

        # 模型选择
        run_model = st.selectbox(
            "模型",
            options=['PatternSearch', 'LSHSearch', 'SAXSearch', 'all'],
            index=3,
            help="选择要运行的模型"
        )

        # ── PatternSearch 参数 ──
        if run_model in ('PatternSearch', 'all'):
            with st.expander("PatternSearch 参数", expanded=False):
                run_top_k = st.number_input(
                    "top_k（近邻数）", min_value=1, max_value=100,
                    value=5, step=1, help="取最近邻的数量"
                )
                run_weighted_ps = st.checkbox("weighted（逆距离加权）", value=True)

        # ── LSHSearch 参数 ──
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

        # ── SAXSearch 参数 ──
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

        # ── 全局参数（手填，支持 TSLib 常用非预设值）────────────────
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
        run_mean_shift = st.checkbox(
            "启用 Mean-Shift（DLinear-style）", value=False,
            help="训练和推理时均减去输入序列均值"
        )
        run_gpu = st.checkbox("启用 GPU 加速", value=False)

        submitted = st.form_submit_button(
            "▶️ 运行实验",
            type="primary",
            use_container_width=True
        )

        # ── 任务三（核心）：提交后通过 subprocess.Popen 非阻塞启动 ──
        if submitted:
            import subprocess
            import sys

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            run_py = os.path.join(project_root, 'run.py')

            cmd = [
                sys.executable,
                run_py,
                "--model", run_model,
                "--seq_len", str(int(run_seq_len)),
                "--pred_len", str(int(run_pred_len)),
            ]
            if run_gpu:
                cmd.append("--use_gpu")
            if run_mean_shift:
                cmd.append("--mean_shift")

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

            # 实时 Log 容器（不阻塞 UI）
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

                spinner_col, log_col = st.columns([1, 3])
                with spinner_col:
                    spinner_placeholder = st.empty()

                with log_col:
                    spinner_placeholder = st.empty()

                # 逐行读取直到进程结束
                for raw_line in iter(process.stdout.readline, ''):
                    if not raw_line:
                        break
                    line = raw_line.rstrip()
                    log_lines.append(line)
                    # 保留最近 200 行避免内存膨胀
                    if len(log_lines) > 200:
                        log_lines = log_lines[-200:]
                    # 实时追加到页面
                    spinner_placeholder.info("⏳ 实验运行中（见右侧日志）...")
                    with log_placeholder.container():
                        st.code("\n".join(log_lines[-80:]), language=None, height=320)

                process.wait()

                with log_placeholder.container():
                    if process.returncode == 0:
                        spinner_placeholder.success("✅ 实验完成！")
                        st.success("✅ 实验完成！正在刷新页面...")
                        st.rerun()
                    else:
                        spinner_placeholder.error(f"❌ 失败 (exit {process.returncode})")
                        st.error(f"❌ 实验失败（exit {process.returncode}）")
                        st.code("\n".join(log_lines[-50:]), language=None, height=200)
            except Exception as ex:
                st.error(f"❌ 启动失败: {ex}")

    # ── 以下：已有实验结果的可视化选择器 ─────────────────────────────
    log_data = load_experiment_log(output_dir)

    if log_data is None:
        st.sidebar.warning("⚠️ 未找到实验日志文件，请先运行实验")
        return {
            'output_dir': output_dir,
            'log_data': None,
            'selected_model': None,
            'selected_pred_len': None,
            'selected_seq_len': None,
            'selected_sample_id': 0,
            'experiments': []
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
            'output_dir': output_dir,
            'log_data': log_data,
            'selected_model': None,
            'selected_pred_len': None,
            'selected_seq_len': None,
            'selected_sample_id': 0,
            'experiments': experiments
        }

    selected_model = st.sidebar.selectbox(
        "选择模型",
        options=available_models,
        index=0,
        help="选择要查看的模型"
    )

    pred_lens = sorted(list(set(
        exp['config']['pred_len']
        for exp in experiments
        if exp['config']['model_name'] == selected_model and exp['status'] == 'success'
    )))

    selected_pred_len = st.sidebar.selectbox(
        "预测长度 (pred_len)",
        options=pred_lens,
        index=0,
        help="选择预测序列长度"
    )

    seq_lens = sorted(list(set(
        exp['config']['seq_len']
        for exp in experiments
        if exp['config']['model_name'] == selected_model
        and exp['config']['pred_len'] == selected_pred_len
        and exp['status'] == 'success'
    )))

    selected_seq_len = st.sidebar.selectbox(
        "序列长度 (seq_len)",
        options=seq_lens,
        index=0,
        help="选择输入序列长度"
    )

    max_samples = min(100, len(experiments) * 5)
    selected_sample_id = st.sidebar.slider(
        "样本 ID",
        min_value=0,
        max_value=max(max_samples - 1, 0),
        value=0,
        help="选择要查看的样本索引 (0-99)"
    )

    return {
        'output_dir': output_dir,
        'log_data': log_data,
        'selected_model': selected_model,
        'selected_pred_len': selected_pred_len,
        'selected_seq_len': selected_seq_len,
        'selected_sample_id': selected_sample_id,
        'experiments': experiments
    }


def render_metrics_comparison(log_data: Dict[str, Any]):
    """
    渲染宏观指标对比部分

    Args:
        log_data: 实验日志数据
    """
    st.markdown("---")
    st.markdown("## 📊 宏观指标对比")

    experiments = log_data.get('experiments', [])
    success_exps = [e for e in experiments if e['status'] == 'success']

    if not success_exps:
        st.warning("没有成功的实验结果")
        return

    # 构建DataFrame
    df_data = []
    for exp in success_exps:
        row = {
            'Model': exp['config']['model_name'],
            'seq_len': exp['config']['seq_len'],
            'pred_len': exp['config']['pred_len'],
            'MAE': exp['metrics'].get('MAE', 0),
            'MSE': exp['metrics'].get('MSE', 0),
            'RMSE': exp['metrics'].get('RMSE', 0),
            'MAPE': exp['metrics'].get('MAPE', 0),
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # 聚合计算平均值
    agg_df = df.groupby(['Model', 'pred_len']).agg({
        'MAE': 'mean',
        'MSE': 'mean',
        'RMSE': 'mean',
        'MAPE': 'mean'
    }).reset_index()

    # 指标选择
    col1, col2 = st.columns(2)

    with col1:
        # MAE柱状图
        st.markdown("### 📉 MAE 对比")

        fig_mae = px.bar(
            agg_df,
            x='pred_len',
            y='MAE',
            color='Model',
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
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            xaxis=dict(tickmode='array', tickvals=[24, 48, 96]),
            height=400
        )

        fig_mae.update_traces(
            texttemplate='%{y:.4f}',
            textposition='outside'
        )

        st.plotly_chart(fig_mae, width="stretch")

    with col2:
        # MSE柱状图
        st.markdown("### 📉 MSE 对比")

        fig_mse = px.bar(
            agg_df,
            x='pred_len',
            y='MSE',
            color='Model',
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
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            xaxis=dict(tickmode='array', tickvals=[24, 48, 96]),
            height=400
        )

        fig_mse.update_traces(
            texttemplate='%{y:.4f}',
            textposition='outside'
        )

        st.plotly_chart(fig_mse, width="stretch")

    # MAPE柱状图
    st.markdown("### 📈 MAPE 对比")

    fig_mape = px.bar(
        agg_df,
        x='pred_len',
        y='MAPE',
        color='Model',
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
        height=350
    )

    fig_mape.update_traces(
        texttemplate='%{y:.2f}%',
        textposition='outside'
    )

    st.plotly_chart(fig_mape, width="stretch")


def render_waveform_comparison(
    config: Dict[str, Any],
    experiments: Optional[List[Dict[str, Any]]] = None
):
    """
    渲染微观波形对比

    任务四（指标）：直接读取 JSON 里的 MAE/MSE/RMSE/MAPE/CORR，
                   用 cols = st.columns(5) 紧凑排列，绝不重新计算。

    任务四（波形）：读取 preview['history']，X 轴从 -seq_len 到 pred_len-1，
                   [-seq_len, -1] 画历史真实，[0, pred_len-1] 画未来真实与预测，
                   x=0 画垂直虚线分隔过去与未来。
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

    if model is None or pred_len is None or seq_len is None:
        st.info("💡 请从侧边栏选择模型和参数后查看波形。")
        return

    st.markdown(f"""
    <div class="info-panel">
        <strong>当前配置:</strong> {model} | seq_len={seq_len} | pred_len={pred_len} | 样本 #{sample_id}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── 任务四（指标5列）：从 JSON 直接读取，不重新计算 ───────────────
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

    cols = st.columns(5)
    metric_labels = ['MAE', 'MSE', 'RMSE', 'MAPE', 'CORR']
    metric_values = [mae, mse, rmse, mape, corr]
    metric_colors = [
        'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
        'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    ]

    for c, label, val, color in zip(cols, metric_labels, metric_values, metric_colors):
        with c:
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 10px;
                        color: white; text-align: center;">
                <div style="font-size: 0.85rem; opacity: 0.9;">{label}</div>
                <div style="font-size: 1.6rem; font-weight: bold; margin-top: 0.3rem;">{val:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── 尝试从 JSON preview 读取连贯波形数据 ─────────────────────────
    preview = None
    if matching_exp and 'preview' in matching_exp:
        preview = matching_exp['preview']

    if preview is not None and 'history' in preview and 'trues' in preview and 'preds' in preview:
        history_list = preview['history']      # list of list（展平的 2D）
        true_list = preview['trues']           # list of list
        pred_list = preview['preds']           # list of list

        if (sample_id < len(history_list)
            and sample_id < len(true_list)
            and sample_id < len(pred_list)):

            # 恢复数据（从展平的 2D 恢复）
            hist_raw = np.array(history_list[sample_id])
            true_raw = np.array(true_list[sample_id])
            pred_raw = np.array(pred_list[sample_id])

            # 推算 n_features：从 true_raw 的长度与 pred_len 推断
            true_n_feat = 1 if true_raw.ndim == 1 else true_raw.shape[-1]

            # 多变量取第 0 特征
            if true_n_feat > 1:
                hist_flat = hist_raw[:(len(hist_raw) // true_n_feat) * true_n_feat]
                hist_sample = hist_flat.reshape(-1, true_n_feat)[:, 0]
                true_flat = true_raw[:(len(true_raw) // true_n_feat) * true_n_feat]
                true_sample = true_flat.reshape(-1, true_n_feat)[:, 0]
                pred_flat = pred_raw[:(len(pred_raw) // true_n_feat) * true_n_feat]
                pred_sample = pred_flat.reshape(-1, true_n_feat)[:, 0]
            else:
                hist_sample = hist_raw.flatten()
                true_sample = true_raw.flatten()
                pred_sample = pred_raw.flatten()

            # ── 连贯波形：X 轴从 -seq_len 到 pred_len-1 ────────────
            st.markdown(f"### 📈 连贯波形（样本 #{sample_id}）")

            fig = go.Figure()

            # 左侧：历史真实（[-seq_len, -1]，淡色背景）
            hist_x = list(range(-seq_len, 0))
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_sample.tolist(), mode='lines',
                name='历史真实 (History)',
                line=dict(color='#94C8D8', width=1.8), opacity=0.7,
                hovertemplate='时间: %{x}<br>历史值: %{y:.4f}<extra></extra>'
            ))

            # 右侧：未来真实（[0, pred_len-1]）
            future_x = list(range(0, len(true_sample)))
            fig.add_trace(go.Scatter(
                x=future_x, y=true_sample.tolist(), mode='lines+markers',
                name='未来真实 (Ground Truth)',
                line=dict(color='#2E86AB', width=2.5),
                marker=dict(size=6, symbol='circle'),
                hovertemplate='时间: %{x}<br>真实值: %{y:.4f}<extra></extra>'
            ))

            # 右侧：模型预测（红色虚线）
            fig.add_trace(go.Scatter(
                x=future_x, y=pred_sample.tolist(), mode='lines+markers',
                name='模型预测 (Prediction)',
                line=dict(color='#E94F37', width=2.5, dash='dash'),
                marker=dict(size=6, symbol='square'),
                hovertemplate='时间: %{x}<br>预测值: %{y:.4f}<extra></extra>'
            ))

            # x=0 垂直虚线（历史/未来分界线）
            fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1.8)

            # 误差区域填充
            fig.add_trace(go.Scatter(
                x=future_x + future_x[::-1],
                y=(pred_sample - true_sample).tolist() + [0] * len(future_x),
                fill='toself', fillcolor='rgba(233, 79, 55, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                name='预测误差带', showlegend=True,
                hovertemplate='时间: %{x}<br>误差: %{y:.4f}<extra></extra>'
            ))

            total_x_range = seq_len + len(true_sample)
            fig.update_layout(
                title=f'{model} | seq={seq_len} | pred={len(true_sample)} | '
                      f'MAE={mae:.4f} | MSE={mse:.4f}',
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
                )
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

            st.info(
                f"📌 **历史上下文说明**：左侧（X < 0）为测试样本的输入历史序列（长度 {seq_len}），"
                "右侧（X ≥ 0）为预测区间。灰色垂直虚线（X=0）为历史与未来的分界线。"
            )

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

                    t_raw = np.array(true_list[idx])
                    p_raw = np.array(pred_list[idx])

                    if true_n_feat > 1:
                        t_flat = t_raw[:(len(t_raw) // true_n_feat) * true_n_feat]
                        t = t_flat.reshape(-1, true_n_feat)[:, 0]
                        p_flat = p_raw[:(len(p_raw) // true_n_feat) * true_n_feat]
                        p = p_flat.reshape(-1, true_n_feat)[:, 0]
                    else:
                        t = t_raw.flatten()
                        p = p_raw.flatten()

                    offset = i * (len(t) + 5)
                    fig_multi.add_trace(go.Scatter(
                        x=[x + offset for x in range(len(t))], y=t.tolist(),
                        mode='lines', name=f'真实 #{idx}',
                        line=dict(color=colors_true[i], width=2), showlegend=True
                    ))
                    fig_multi.add_trace(go.Scatter(
                        x=[x + offset for x in range(len(p))], y=p.tolist(),
                        mode='lines', name=f'预测 #{idx}',
                        line=dict(color=colors_pred[i], width=2, dash='dash'), showlegend=True
                    ))

            fig_multi.update_layout(
                title='连续 5 个样本的预测对比',
                xaxis_title='时间步（带偏移）',
                yaxis_title='值',
                template='plotly_white',
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.15, xanchor='right', x=1)
            )
            st.plotly_chart(fig_multi, width="stretch")

        else:
            st.error(f"样本 ID {sample_id} 超出范围")

    else:
        # 没有 JSON preview：尝试从 .npy 加载
        config_params = matching_exp['config'] if matching_exp else None
        preds, trues, x_test = load_predictions(
            output_dir, model, seq_len, pred_len, config_params
        )

        if preds is None or trues is None:
            st.info("💡 提示: 预测结果文件不存在，请确保已运行相应实验。")
            st.markdown("### 📋 示例波形展示（模拟数据）")

            n_timesteps = pred_len
            x_axis = list(range(n_timesteps))
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
                x=x_axis, y=true_wave, mode='lines+markers',
                name='真实值 (Ground Truth)',
                line=dict(color='#2E86AB', width=2), marker=dict(size=6),
                hovertemplate='时间点: %{x}<br>真实值: %{y:.4f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=x_axis, y=pred_wave, mode='lines+markers',
                name='预测值 (Prediction)',
                line=dict(color='#E94F37', width=2, dash='dash'), marker=dict(size=6),
                hovertemplate='时间点: %{x}<br>预测值: %{y:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=f'{model} 模型预测效果（示例数据）',
                xaxis_title='时间步 (Timestep)',
                yaxis_title='值 (Value)',
                template='plotly_white',
                hovermode='x unified',
                height=450,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, width="stretch")
        else:
            # 从 .npy 加载时的连贯波形渲染
            if sample_id < len(preds) and sample_id < len(trues):
                pred_s = preds[sample_id]
                true_s = trues[sample_id]
                if pred_s.ndim > 1:
                    pred_s = pred_s[:, 0]
                if true_s.ndim > 1:
                    true_s = true_s[:, 0]

                hist_s = None
                if x_test is not None and sample_id < len(x_test):
                    h = x_test[sample_id]
                    hist_s = h[:, 0] if h.ndim > 1 else h.flatten()

                st.markdown(f"### 📈 预测结果波形（样本 #{sample_id}）")
                fig = go.Figure()

                if hist_s is not None:
                    fig.add_trace(go.Scatter(
                        x=list(range(-seq_len, 0)), y=hist_s.tolist(), mode='lines',
                        name='历史真实 (History)',
                        line=dict(color='#94C8D8', width=1.8), opacity=0.7,
                        hovertemplate='时间: %{x}<br>历史值: %{y:.4f}<extra></extra>'
                    ))

                fx = list(range(0, len(true_s)))
                fig.add_trace(go.Scatter(
                    x=fx, y=true_s.tolist(), mode='lines+markers',
                    name='未来真实 (Ground Truth)',
                    line=dict(color='#2E86AB', width=2.5),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate='时间: %{x}<br>真实值: %{y:.4f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=fx, y=pred_s.tolist(), mode='lines+markers',
                    name='模型预测 (Prediction)',
                    line=dict(color='#E94F37', width=2.5, dash='dash'),
                    marker=dict(size=6, symbol='square'),
                    hovertemplate='时间: %{x}<br>预测值: %{y:.4f}<extra></extra>'
                ))
                fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1.8)

                mae_v = float(np.mean(np.abs(pred_s - true_s)))
                mse_v = float(np.mean((pred_s - true_s) ** 2))

                fig.add_trace(go.Scatter(
                    x=fx + fx[::-1],
                    y=(pred_s - true_s).tolist() + [0] * len(fx),
                    fill='toself', fillcolor='rgba(233, 79, 55, 0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='预测误差带', showlegend=True,
                    hovertemplate='时间: %{x}<br>误差: %{y:.4f}<extra></extra>'
                ))

                total_x_range = seq_len + len(true_s)
                fig.update_layout(
                    title=f'{model} | seq={seq_len} | pred={len(true_s)} | '
                          f'MAE={mae_v:.4f} | MSE={mse_v:.4f}',
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
                    )
                )
                st.plotly_chart(fig, width="stretch")

                ss1, ss2, ss3, ss4 = st.columns(4)
                with ss1:
                    st.metric("真实均值", f"{float(np.mean(true_s)):.4f}")
                with ss2:
                    st.metric("预测均值", f"{float(np.mean(pred_s)):.4f}")
                with ss3:
                    st.metric("MAE", f"{mae_v:.4f}")
                with ss4:
                    st.metric("MSE", f"{mse_v:.4f}")

                st.info(
                    f"📌 **历史上下文说明**：左侧（X < 0）为测试样本的输入历史序列（长度 {seq_len}），"
                    "右侧（X ≥ 0）为预测区间。灰色垂直虚线（X=0）为历史与未来的分界线。"
                )
            else:
                st.error(f"样本 ID {sample_id} 超出范围")


def render_model_introduction():
    """渲染模型介绍"""
    st.markdown("---")
    st.markdown("## 📚 模型介绍")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### PatternSearch
        **基于 KD-Tree 的 KNN 检索**

        - 使用欧氏距离度量相似性
        - 逆距离加权平均
        - 精确近邻搜索
        """)

    with col2:
        st.markdown("""
        ### LSHSearch
        **局部敏感哈希检索**

        - 随机投影生成哈希码
        - 多哈希表增加召回率
        - 汉明距离容忍匹配
        """)

    with col3:
        st.markdown("""
        ### SAXSearch
        **符号聚合近似 + 最近邻检索**

        - PAA 降维压缩为 word_size 段
        - 高斯分位数字符化（整数打包）
        - sklearn NearestNeighbors 模糊匹配（PAA 空间）
        """)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    # 渲染头部
    render_header()

    # 渲染侧边栏
    config = render_sidebar()

    # 检查数据可用性
    log_data = config.get('log_data')

    if log_data is None:
        st.warning("⚠️ 请先运行 `python run.py --model all` 生成实验结果（或指定模型）")
        st.markdown("---")
        render_model_introduction()
        return

    # 渲染宏观指标对比
    render_metrics_comparison(log_data)

    # 渲染微观波形对比（显式传入 experiments，避免作用域/缺参导致 NameError）
    render_waveform_comparison(config, experiments=log_data.get('experiments', []))

    # 渲染模型介绍
    render_model_introduction()


if __name__ == '__main__':
    main()
