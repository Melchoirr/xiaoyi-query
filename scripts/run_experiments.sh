#!/usr/bin/env bash
# =====================================================================
# run_experiments.sh - 时序预测基线系统 v4.0 顶级调度脚本
# =====================================================================
# 大道至简重构版：Shell 全权负责参数网格展开 + 并行调度
# Python 端纯粹执行器（run.py），单次调用仅执行单一实验
#
# 设计理念：
#   1. Shell 生成全局 RUN_DIR（时间戳隔离）
#   2. case 语句为每个算法编写独立的超参循环（精确控制）
#   3. 后台任务（&）+ wait 实现 Shell 级并行
#   4. 实验结果追加到 summary_metrics.csv
#   5. 脚本结束时调用 Python 生成超级矩阵图
#
# Usage:
#   ./scripts/run_experiments.sh --parallel --use-gpu
#   ./scripts/run_experiments.sh --model PatternSearch --seq-lens 96 --pred-lens 96
#   ./scripts/run_experiments.sh --models all --dry-run
# =====================================================================

# =====================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    七大算法超参百科全书                               ║
# ║                                                                      ║
# ║  本节详尽列出所有算法的可调参数、默认值、有效范围、以及               ║
# ║  内存占用和性能代价评估。旨在打造"教科书级"调度脚本。                ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# ───────────────────────────────────────────────────────────────────────
# 1. PatternSearch（欧氏距离 KNN 检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: torch.cdist GPU 批量计算欧氏距离，torch.topk 取 Top-K
#   精度:    精确（Exact Nearest Neighbor）
#   时间复杂度: O(n_test * n_train * seq_len)
#
#   可调参数:
#     --top_k (int)
#       描述: 检索的近邻数量
#       默认值: 5
#       有效范围: [1, 20]
#       影响: k↑ → 预测更平滑，但计算量增加；k↓ → 预测更尖锐，可能过拟合
#
#     --weighted (bool)
#       描述: 是否使用逆距离加权平均
#       默认值: true
#       有效值: true / false
#       影响: true → 距离越近的邻居权重越大；false → 等权平均
#
#     --predict_chunk_size (int)
#       描述: 推理时分块大小（样本数）
#       默认值: 4096
#       有效范围: [256, 16384]
#       内存代价: chunk_size * n_train * 4 bytes (float32)
#       性能提示: V100 32GB 推荐 4096-8192
#
#   内存评估 (seq=96, n_feat=7, chunk=4096):
#     显存 ≈ 4096 * 8353 * 96 * 7 * 4B ≈ 9.2 GB (若全驻留)
#     实际: 分块计算，显存占用 ≈ chunk_size * 4B * 2 ≈ 32 MB
#
# ───────────────────────────────────────────────────────────────────────
# 2. LSHSearch（局部敏感哈希近似检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: 随机投影哈希 + 两阶段重排（候选集 → 精确距离）
#   精度:    近似（Approximate Nearest Neighbor）
#   时间复杂度: O(n_test * n_hash_funcs * n_tables)
#
#   可调参数:
#     --n_hash_funcs (int)
#       描述: 每个表的哈希函数数量
#       默认值: 16
#       有效范围: [4, 32]
#       影响: 越多 → 哈希碰撞概率↑，召回率↑，但计算量↑
#
#     --n_tables (int)
#       描述: 哈希表数量
#       默认值: 4
#       有效范围: [1, 8]
#       影响: 越多 → 候选集越大，召回率↑，内存占用↑
#
#     --hamming_radius (int)
#       描述: 汉明距离容忍半径
#       默认值: 2
#       有效范围: [0, 5]
#       影响: 越大 → 候选集越大，召回率↑，但可能引入噪声
#
#     --candidate_cap_per_table (int)
#       描述: 每个表的最大候选数量
#       默认值: 256
#       有效范围: [32, 2048]
#
#     --candidate_cap_total (int)
#       描述: 所有表的候选数量上限
#       默认值: 1024
#       有效范围: [64, 8192]
#
#     --lsh_weighted (bool)
#       描述: 是否使用加权投票
#       默认值: false
#
#   内存评估:
#     内存 ≈ n_train * n_hash_funcs * 8 bytes ≈ 8353 * 16 * 8B ≈ 1 MB
#     极低内存占用是其最大优势
#
# ───────────────────────────────────────────────────────────────────────
# 3. SAXSearch（符号聚合近似检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: PAA(Piecewise Aggregate Approximation) 降维 → 符号化 → NN 检索
#   精度:    模糊近似
#   时间复杂度: O(n_test * word_size * alphabet_size)
#
#   可调参数:
#     --word_size (int)
#       描述: PAA 分段数（也即 SAX 符号串长度）
#       默认值: 8
#       有效范围: [4, 16]
#       影响: 越大 → 表示越精细，但搜索空间指数增长
#       注意: word_size * alphabet_size 应 ≈ seq_len
#
#     --alphabet_size (int)
#       描述: 符号表大小（通常 4-8）
#       默认值: 8
#       有效范围: [2, 8]
#       影响: 越大 → 区分度↑，但需要更多样本才能可靠估计
#
#     --epsilon_threshold (float)
#       描述: 相似度阈值
#       默认值: 1.0
#       有效范围: [0.1, 5.0]
#
#     --bucket_top_k (int)
#       描述: 每个桶取前 k 个候选
#       默认值: 8
#       有效范围: [1, 32]
#
#     --sax_weighted (bool)
#       描述: 是否使用加权检索
#       默认值: true
#
#   内存评估:
#     内存 ≈ n_train * word_size * 1 byte ≈ 8353 * 8B ≈ 67 KB
#     极低内存
#
# ───────────────────────────────────────────────────────────────────────
# 4. DTWSearch（动态时间规整弹性检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: Sakoe-Chiba 约束的累积 DP，计算弹性时间对齐距离
#   精度:    精确（受 Sakoe-Chiba 约束限制）
#   时间复杂度: O(n_test * n_train * seq_len * radius)
#
#   可调参数:
#     --top_k (int)
#       描述: 检索的近邻数量
#       默认值: 5
#       有效范围: [1, 20]
#
#     --dtw_radius (int)
#       描述: Sakoe-Chiba 约束半径
#       默认值: 5
#       有效范围: [1, 20]
#       影响: 越大 → 允许更大时间偏移，但计算量 O(radius) 增长
#       推荐: seq=96 → radius=5-10; seq=192 → radius=10-20
#
#     --weighted (bool)
#       描述: 是否使用逆距离加权
#       默认值: true
#
#     --predict_chunk_size (int)
#       描述: 测试集分块大小
#       默认值: 128 (GPU) / 64 (CPU)
#       有效范围: [32, 512]
#       内存代价: 最重，chunk * n_train * seq_len * 4B
#       示例: chunk=128, n_train=8353, seq=96 → 410 MB
#
#   内存评估 (GPU):
#     每块显存 ≈ chunk * n_train * seq_len * 4B
#     chunk=128, seq=96 → 410 MB
#     chunk=256, seq=96 → 820 MB
#     V100 32GB 可用 chunk=256-512
#
# ───────────────────────────────────────────────────────────────────────
# 5. MatrixProfileSearch（矩阵剖面 Z-Norm 检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: GPU 向量化 Z-Normalized 欧氏距离 + torch.topk
#   精度:    精确子序列匹配
#   时间复杂度: O(n_test * n_train * seq_len)
#
#   可调参数:
#     --top_k (int)
#       描述: 检索的近邻数量
#       默认值: 5
#       有效范围: [1, 20]
#
#     --subsequence_length (int, 可为 None)
#       描述: 子序列长度（None=seq_len）
#       默认值: None (使用 seq_len)
#       影响: 可用于分层匹配
#
#     --mp_normalize (bool)
#       描述: 是否进行 Z-Normalization
#       默认值: true
#       影响: Z-Norm 可消除幅度差异，关注形态相似性
#
#     --mp_chunk_size (int)
#       描述: 测试集分块大小
#       默认值: 4096
#       有效范围: [256, 8192]
#       内存代价: 与 PatternSearch 类似，但每个样本需计算 mean/std
#
#     --mp_train_chunk_size (int)
#       描述: 训练集分块大小
#       默认值: 2048
#       有效范围: [256, 4096]
#
#   内存评估:
#     与 PatternSearch 类似，但需额外存储 mean/std
#     V100 推荐 chunk=4096-8192
#
# ───────────────────────────────────────────────────────────────────────
# 6. TS2VecSearch（深度对比学习表示检索）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: TCN 编码器 + 对比学习预训练 + faiss 向量检索
#   精度:    深度表示近似检索
#   时间复杂度: O(n_epochs * n_train * batch_size * seq_len) + O(n_test * log(n_train))
#
#   可调参数:
#     --hidden_dim (int)
#       描述: 编码器隐向量维度
#       默认值: 64
#       有效范围: [32, 256]
#       影响: 越大 → 表示能力↑，但训练时间和内存↑
#
#     --ts2vec_epochs (int)
#       描述: 对比学习训练轮数
#       默认值: 100
#       有效范围: [20, 200]
#       影响: 越多 → 特征表示越丰富，但收益递减
#       性能提示: V100 epoch=100 约 3-5 分钟
#
#     --ts2vec_batch_size (int)
#       描述: 训练批大小
#       默认值: 1024
#       有效范围: [128, 2048]
#       内存代价: batch_size * seq_len * hidden_dim * 4B * 2 (对比两个视图)
#
#     --ts2vec_lr (float)
#       描述: 学习率
#       默认值: 1e-3
#       有效范围: [1e-5, 1e-2]
#
#     --temperature (float)
#       描述: NT-Xent 温度参数
#       默认值: 0.1
#       有效范围: [0.01, 1.0]
#       影响: 越小 → 越关注难负样本
#
#     --top_k (int)
#       描述: faiss 检索的邻居数量
#       默认值: 5
#
#   内存评估 (V100):
#     显存 ≈ batch_size * seq_len * hidden_dim * 4B * 4 ≈ 1024 * 96 * 64 * 16B ≈ 100 MB
#     faiss 索引 ≈ n_train * hidden_dim * 4B ≈ 8353 * 64 * 4B ≈ 2 MB
#
# ───────────────────────────────────────────────────────────────────────
# 7. RAGSearch（检索增强生成网络）
# ───────────────────────────────────────────────────────────────────────
#   核心方法: Siamese Cross-Attention，Query/Key 共享编码器
#   精度:    端到端学习
#   时间复杂度: O(n_epochs * n_train * batch_size * seq_len * d_model)
#
#   可调参数:
#     --rag_d_model (int)
#       描述: 交叉注意力隐向量维度
#       默认值: 256
#       有效范围: [64, 512]
#       影响: 越大 → 模型容量↑，但训练时间↑
#
#     --rag_n_heads (int)
#       描述: 注意力头数
#       默认值: 8
#       有效范围: [2, 16]
#       注意: d_model 必须能被 n_heads 整除
#
#     --rag_epochs (int)
#       描述: 训练轮数
#       默认值: 100
#       有效范围: [20, 200]
#       性能提示: V100 epoch=100 约 5-8 分钟
#
#     --rag_batch_size (int)
#       描述: 训练批大小
#       默认值: 1024
#       有效范围: [128, 2048]
#
#     --rag_lr (float)
#       描述: 学习率
#       默认值: 1e-3
#       有效范围: [1e-5, 1e-2]
#
#     --rag_weight_decay (float)
#       描述: 权重衰减
#       默认值: 1e-4
#       有效范围: [1e-6, 1e-2]
#
#   内存评估 (V100):
#     显存 ≈ batch_size * seq_len * d_model * 4B * 4 (Q, K, V, attention) ≈ 1024 * 96 * 256 * 64B ≈ 1.5 GB
#     加上梯度 ≈ 3 GB
#
# ───────────────────────────────────────────────────────────────────────
# 全局参数（所有算法共享）
# ───────────────────────────────────────────────────────────────────────
#   --seq_len (int)
#     描述: 输入序列长度（历史窗口）
#     默认值: 96
#     有效范围: [24, 720]
#     推荐: TSLib 标准 [96, 192, 336, 720]
#
#   --pred_len (int)
#     描述: 预测长度（未来窗口）
#     默认值: 96
#     有效范围: [24, 720]
#     推荐: TSLib 标准 [48, 96, 192, 336, 720]
#
#   --revin_type (str)
#     描述: RevIN 归一化类型
#     默认值: none
#     有效值: none / temporal / feature / dual
#     影响:
#       - none: 无归一化
#       - temporal: 时间维度归一化（每个时间步独立）
#       - feature: 特征维度归一化（每个特征独立）
#       - dual: 先特征归一化，再时间归一化（推荐）
#
#   --use_gpu (flag)
#     描述: 是否使用 GPU
#     自动检测: 若 torch.cuda.is_available() 则默认启用
#
# =====================================================================

set -euo pipefail

# ── 项目路径 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

# ── 全局运行目录（Shell 生成，时间戳隔离）─────────────────────────
GLOBAL_RUN_DIR="${PROJECT_DIR}/results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GLOBAL_RUN_DIR/logs"

# ── 默认参数 ──────────────────────────────────────────────────────
MODELS="${MODELS:-all}"
SEQ_LENS="${SEQ_LENS:-96}"
PRED_LENS="${PRED_LENS:-96}"
REVIN_TYPES="${REVIN_TYPES:-dual}"
PARALLEL="${PARALLEL:-true}"
GPU_FLAG=""
DRY_RUN=false
MAX_PARALLEL_JOBS=8

# ── 参数解析 ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODELS="$2"; shift 2 ;;
        --models)         MODELS="$2"; shift 2 ;;
        --seq-lens)       SEQ_LENS="$2"; shift 2 ;;
        --pred-lens)      PRED_LENS="$2"; shift 2 ;;
        --revin-types)    REVIN_TYPES="$2"; shift 2 ;;
        --parallel)       PARALLEL="true"; shift ;;
        --no-parallel)    PARALLEL="false"; shift ;;
        --use-gpu)        GPU_FLAG="--use_gpu"; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --max-jobs)       MAX_PARALLEL_JOBS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── 展开 all 为完整模型列表（必须在 IFS 解析之前）─────────────────
if [[ "$MODELS" == "all" ]]; then
    MODELS="PatternSearch,LSHSearch,SAXSearch,DTWSearch,MatrixProfileSearch,TS2VecSearch,RAGSearch"
fi

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra SEQ_ARR <<< "$SEQ_LENS"
IFS=',' read -ra PRED_ARR <<< "$PRED_LENS"
IFS=',' read -ra REVIN_ARR <<< "$REVIN_TYPES"

# ── 头部信息 ──────────────────────────────────────────────────────
echo "================================================================================"
echo "  Time-Series Forecasting Baseline v4.0 - 大道至简重构版"
echo "================================================================================"
echo "  GLOBAL_RUN_DIR: $GLOBAL_RUN_DIR"
echo "  Models:        ${MODEL_ARR[*]}"
echo "  seq_len:       ${SEQ_ARR[*]}"
echo "  pred_len:      ${PRED_ARR[*]}"
echo "  revin_type:    ${REVIN_ARR[*]}"
echo "  GPU:           ${GPU_FLAG:-(auto-detect)}"
echo "  Parallel:       $PARALLEL (max jobs: $MAX_PARALLEL_JOBS)"
echo "  Dry Run:       $DRY_RUN"
echo "================================================================================"

# ── 计数器 ───────────────────────────────────────────────────────
TOTAL_JOBS=0
RUNNING_JOBS=0
declare -A PIDS
declare -A JOB_STATUS
PASS=0
FAIL=0

# ── 并发控制辅助函数 ──────────────────────────────────────────────
wait_for_slot() {
    while true; do
        # 统计当前运行的任务数
        local active=0
        for pid in "${!PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                active=$((active + 1))
            fi
        done
        if [[ $active -lt $MAX_PARALLEL_JOBS ]]; then
            break
        fi
        sleep 1
    done
}

check_jobs() {
    # 检查已完成的任务
    for pid in "${!PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            local exit_code=$?
            JOB_STATUS[$pid]=$exit_code
            if [[ $exit_code -eq 0 ]]; then
                PASS=$((PASS + 1))
            else
                FAIL=$((FAIL + 1))
            fi
            unset PIDS[$pid]
        fi
    done
}

# ── 通用运行函数 ─────────────────────────────────────────────────
run_experiment() {
    local model="$1"
    shift
    local seq="$1"
    shift
    local pred="$1"
    shift
    local revin="$1"
    shift
    local extra_args="$@"

    # 生成 exp_id 后缀
    local exp_suffix=""
    case "$model" in
        PatternSearch)       exp_suffix="${extra_args}" ;;
        LSHSearch)           exp_suffix="${extra_args}" ;;
        SAXSearch)           exp_suffix="${extra_args}" ;;
        DTWSearch)           exp_suffix="${extra_args}" ;;
        MatrixProfileSearch) exp_suffix="${extra_args}" ;;
        TS2VecSearch)        exp_suffix="${extra_args}" ;;
        RAGSearch)           exp_suffix="${extra_args}" ;;
    esac

    local cmd=(
        "$PYTHON" "$PROJECT_DIR/run.py"
        --model "$model"
        --seq_len "$seq"
        --pred_len "$pred"
        --revin_type "$revin"
        --run_dir "$GLOBAL_RUN_DIR"
        $extra_args
        ${GPU_FLAG:+"$GPU_FLAG"}
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] ${cmd[*]}"
        return 0
    fi

    # 执行命令
    "${cmd[@]}" &
    local pid=$!
    PIDS[$pid]=1
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    echo "  [$TOTAL_JOBS] Started: $model seq=$seq pred=$pred revin=$revin (PID=$pid)"
}

# ── 主调度循环 ────────────────────────────────────────────────────
echo ""
echo "Starting experiment scheduling..."
echo ""

for model in "${MODEL_ARR[@]}"; do
    echo "=== Processing model: $model ==="

    case "$model" in

        # ══════════════════════════════════════════════════════════════
        # PatternSearch: top_k + weighted
        # ══════════════════════════════════════════════════════════════
        PatternSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for top_k in 3 5 10; do
                            for weighted in true false; do
                                if [[ "$PARALLEL" == "true" ]]; then
                                    wait_for_slot
                                fi
                                run_experiment "$model" "$seq" "$pred" "$revin" \
                                    "--top_k $top_k --weighted $weighted"
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # LSHSearch: 哈希参数
        # ══════════════════════════════════════════════════════════════
        LSHSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for n_funcs in 8 16 24; do
                            for n_tables in 2 4 6; do
                                if [[ "$PARALLEL" == "true" ]]; then
                                    wait_for_slot
                                fi
                                run_experiment "$model" "$seq" "$pred" "$revin" \
                                    "--n_hash_funcs $n_funcs --n_tables $n_tables"
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # SAXSearch: PAA 参数
        # ══════════════════════════════════════════════════════════════
        SAXSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for word_size in 4 8 12; do
                            for alpha_size in 4 8; do
                                if [[ "$PARALLEL" == "true" ]]; then
                                    wait_for_slot
                                fi
                                run_experiment "$model" "$seq" "$pred" "$revin" \
                                    "--word_size $word_size --alphabet_size $alpha_size"
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # DTWSearch: top_k + dtw_radius
        # ══════════════════════════════════════════════════════════════
        DTWSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for top_k in 3 5 10; do
                            for radius in 3 5 10; do
                                if [[ "$PARALLEL" == "true" ]]; then
                                    wait_for_slot
                                fi
                                run_experiment "$model" "$seq" "$pred" "$revin" \
                                    "--top_k $top_k --dtw_radius $radius"
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # MatrixProfileSearch: top_k + normalize
        # ══════════════════════════════════════════════════════════════
        MatrixProfileSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for top_k in 3 5 10; do
                            for normalize in true false; do
                                if [[ "$PARALLEL" == "true" ]]; then
                                    wait_for_slot
                                fi
                                run_experiment "$model" "$seq" "$pred" "$revin" \
                                    "--top_k $top_k --mp_normalize $normalize"
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # TS2VecSearch: hidden_dim + epochs + top_k
        # ══════════════════════════════════════════════════════════════
        TS2VecSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for hidden in 64 128; do
                            for epochs in 50 100; do
                                for top_k in 5 10; do
                                    if [[ "$PARALLEL" == "true" ]]; then
                                        wait_for_slot
                                    fi
                                    run_experiment "$model" "$seq" "$pred" "$revin" \
                                        "--hidden_dim $hidden --ts2vec_epochs $epochs --top_k $top_k"
                                done
                            done
                        done
                    done
                done
            done
            ;;

        # ══════════════════════════════════════════════════════════════
        # RAGSearch: d_model + n_heads + epochs（无 top_k）
        # ══════════════════════════════════════════════════════════════
        RAGSearch)
            for seq in "${SEQ_ARR[@]}"; do
                for pred in "${PRED_ARR[@]}"; do
                    for revin in "${REVIN_ARR[@]}"; do
                        for d_model in 128 256; do
                            for n_heads in 4 8; do
                                for epochs in 50 100; do
                                    if [[ "$PARALLEL" == "true" ]]; then
                                        wait_for_slot
                                    fi
                                    run_experiment "$model" "$seq" "$pred" "$revin" \
                                        "--rag_d_model $d_model --rag_n_heads $n_heads --rag_epochs $epochs"
                                done
                            done
                        done
                    done
                done
            done
            ;;

        *)
            echo "Unknown model: $model, skipping..."
            ;;
    esac

    # 每处理完一个模型，检查并等待所有任务完成
    if [[ "$PARALLEL" == "false" ]]; then
        # 串行模式：等待当前模型的所有任务完成
        wait
    else
        # 并行模式：持续检查
        while [[ ${#PIDS[@]} -gt 0 ]]; do
            check_jobs
            sleep 2
        done
    fi

    echo "=== $model completed ==="
    echo ""

done

# ── 等待所有任务完成 ──────────────────────────────────────────────
echo ""
echo "Waiting for all jobs to complete..."

while [[ ${#PIDS[@]} -gt 0 ]]; do
    check_jobs
    sleep 2
    echo "  Progress: ${#PIDS[@]} jobs running | PASS=$PASS | FAIL=$FAIL"
done

# ── 生成汇总图表 ──────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "All experiments completed!"
echo "PASS: $PASS | FAIL: $FAIL | TOTAL: $TOTAL_JOBS"
echo "================================================================================"
echo ""

if [[ "$DRY_RUN" == "false" ]]; then
    echo "Generating summary visualizations..."

    # 调用 Python 生成超级矩阵图 + 跨模型对比图
    "$PYTHON" - "$GLOBAL_RUN_DIR" <<'PYEOF'
import sys
import os

run_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

try:
    from plotting import generate_all_plots, plot_cross_model_comparison
    print(f"[Plotting] Generating all plots for: {run_dir}")
    generate_all_plots(run_dir)
    print("[Plotting] Super matrix, retrieval plots, and paper-level comparison generated!")

except ImportError as e:
    print(f"[Plotting] Import error: {e}")
except Exception as e:
    print(f"[Plotting] Error: {e}")

print(f"\nResults directory: {run_dir}")
print(f"Summary CSV: {os.path.join(run_dir, 'summary_metrics.csv')}")
print(f"Logs: {os.path.join(run_dir, 'logs')}")
PYEOF

    echo ""
    echo "================================================================================"
    echo "  Run Summary"
    echo "================================================================================"
    echo "  Run Directory: $GLOBAL_RUN_DIR"
    echo "  Summary CSV:   $GLOBAL_RUN_DIR/summary_metrics.csv"
    echo "  Super Matrix:  $GLOBAL_RUN_DIR/super_comparison_matrix.png"
    echo "  Model Ranking: $GLOBAL_RUN_DIR/model_ranking_bar.png"
    echo "  Cross-Model:   $GLOBAL_RUN_DIR/paper_level_comparison.png"
    echo "  Logs:          $GLOBAL_RUN_DIR/logs/"
    echo "================================================================================"
fi

echo ""
echo "Done!"
