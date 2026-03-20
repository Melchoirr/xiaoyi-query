"""
ETT数据摄入脚本
将ETT时序数据通过滑动窗口切分后，批量注入Qdrant向量数据库

注意：摄入时请先停止 FastAPI 服务（Qdrant 不支持并发写入）。
摄入完成后再启动服务即可。

用法:
    python scripts/ingest_ett.py                       # 交互式选择
    python scripts/ingest_ett.py --file ETTm1         # 指定数据集
    python scripts/ingest_ett.py --file ETTm1 --column OT  # 指定列
    python scripts/ingest_ett.py --file ETTm1 --column OT --history-len 100 --future-len 48
    python scripts/ingest_ett.py --dry-run             # 仅预览不写入
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from core.processor import TSProcessor
from components.encoder import ONNXEncoder
from components.retriever import QdrantRetriever


COLLECTION_NAME = "time_series_rag"
INPUT_LENGTH = 100
STORAGE_PATH = "./qdrant_data"
ONNX_PATH = "encoder_v1.onnx"
DEFAULT_FUTURE_LENGTH = 48
BATCH_SIZE = 500


# ----------------------------------------------------------------------
# 时间特征提取
# ----------------------------------------------------------------------

def extract_time_features(ts: datetime) -> dict:
    """
    从时间戳中提取结构化时间特征，构造 Qdrant Payload 过滤字段。

    规则:
        - month:       月份 (1-12)
        - hour:        小时 (0-23)
        - is_weekend:  是否周末 (bool)
        - time_of_day: 时段分类
                        "night"     00:00-06:00
                        "morning"   06:00-12:00
                        "afternoon" 12:00-18:00
                        "evening"   18:00-24:00

    Returns:
        dict: {"month": int, "hour": int, "is_weekend": bool, "time_of_day": str}
    """
    hour = ts.hour
    is_weekend = ts.weekday() >= 5  # Saturday=5, Sunday=6

    if 0 <= hour < 6:
        time_of_day = "night"
    elif 6 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 18:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    return {
        "month": ts.month,
        "hour": hour,
        "is_weekend": is_weekend,
        "time_of_day": time_of_day,
    }


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------

def load_ett_data(csv_path: str) -> pd.DataFrame:
    """加载ETT CSV数据"""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    numeric_cols = [c for c in df.columns if c != "date"]
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  可用列: {numeric_cols}")
    return df, numeric_cols


# ----------------------------------------------------------------------
# 滑动窗口切分（携带时间戳）
# ----------------------------------------------------------------------

def sliding_window_sequences(
    df: pd.DataFrame,
    column: str,
    history_len: int,
    future_len: int,
) -> list:
    """
    滑动窗口切分时序数据，同时携带时间戳用于特征提取。

    窗口结构:
        history_x: [t, t+1, ..., t+history_len-1]  长度 history_len
        future_y:  [t+history_len, ..., t+history_len+future_len-1]  长度 future_len

    参数:
        df:          包含 "date" 列和目标列的 DataFrame
        column:      要切分的数值列名
        history_len: 历史窗口长度
        future_len:  未来窗口长度

    Returns:
        List of (history_x, future_y, row_idx, timestamp) tuples
        - history_x:  np.ndarray, 历史序列
        - future_y:   np.ndarray, 未来序列
        - row_idx:    int, 切片起始行索引 (history 起始位置)
        - timestamp:  pd.Timestamp, 切片对应的时间点 (取 future 窗口起始时间)
    """
    total_len = history_len + future_len
    dates = df["date"].values
    values = df[column].values.astype(np.float64)

    sequences = []
    for i in range(len(df) - total_len + 1):
        history = values[i : i + history_len]
        future = values[i + history_len : i + total_len]
        # 时间点取 future 窗口的起始时间，代表"接下来"这一段发生的时间
        ts = pd.Timestamp(dates[i + history_len])
        sequences.append((history, future, i, ts))

    return sequences


def create_ingestion_batches(sequences: list, batch_size: int) -> list:
    """将序列列表拆分为批次"""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i : i + batch_size])
    return batches


# ----------------------------------------------------------------------
# 单列摄入
# ----------------------------------------------------------------------

def ingest_column(
    df: pd.DataFrame,
    column: str,
    history_len: int,
    future_len: int,
    batch_size: int,
    dry_run: bool,
    processor: TSProcessor,
    encoder: ONNXEncoder,
    retriever: QdrantRetriever,
    offset: int = 0,
) -> int:
    """
    摄入单列数据，将滑动窗口切片编码后批量写入 Qdrant。

    Payload Schema (新增时间特征):
        {
            "source":       str,   # 列名
            "future_y":     list,   # 未来序列（原始值）
            "mu":           float,  # Z-Score 均值
            "sigma":        float,  # Z-Score 标准差
            "history_x":    list,   # 历史序列（原始值）
            # ---- 时间特征 (用于 Qdrant Filtering) ----
            "month":        int,    # 1-12
            "hour":         int,    # 0-23
            "is_weekend":   bool,
            "time_of_day":  str,    # "night" | "morning" | "afternoon" | "evening"
        }
    """
    print(f"\n  列名: {column}")
    print(f"  序列长度: {len(df)}")

    nan_count = df[column].isna().sum()
    if nan_count > 0:
        print(f"  警告: 检测到 {nan_count} 个缺失值，将跳过含NaN的窗口")

    sequences = sliding_window_sequences(df, column, history_len, future_len)
    print(f"  滑动窗口: history={history_len}, future={future_len}")
    print(f"  可生成样本数: {len(sequences)}")

    if len(sequences) == 0:
        print(f"  跳过: 数据不足以生成样本 (需要 {history_len + future_len} 个点)")
        return 0

    if dry_run:
        print(f"  [Dry Run] 前3个样本预览:")
        for hx, fy, idx, ts in sequences[:3]:
            tf = extract_time_features(ts)
            print(
                f"    窗口[{idx}] @ {ts.strftime('%Y-%m-%d %H:%M')}  "
                f"history_x=[{hx.min():.2f}, {hx.max():.2f}]  "
                f"future_y=[{fy.min():.2f}, {fy.max():.2f}]  "
                f"[month={tf['month']}, hour={tf['hour']}, "
                f"weekend={tf['is_weekend']}, tod={tf['time_of_day']}]"
            )
        return len(sequences)

    batches = create_ingestion_batches(sequences, batch_size)
    total_ingested = 0

    for batch_idx, batch in enumerate(batches):
        vectors = []
        payloads = []

        for history_x, future_y, _, ts in batch:
            normalized, mu, sigma = processor.normalize(history_x.tolist())
            embedding = encoder.encode(normalized)
            time_features = extract_time_features(ts)

            payload = {
                "source": column,
                "future_y": future_y.tolist(),
                "mu": float(mu),
                "sigma": float(sigma),
                "history_x": history_x.tolist(),
                # 时间特征
                "month": time_features["month"],
                "hour": time_features["hour"],
                "is_weekend": time_features["is_weekend"],
                "time_of_day": time_features["time_of_day"],
            }
            vectors.append(embedding)
            payloads.append(payload)

        vector_ids = retriever.ingest_batch(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            payloads=payloads,
        )

        total_ingested += len(batch)
        progress = batch_idx + 1
        print(
            f"    批次 {progress}/{len(batches)}: "
            f"+{len(batch)} 条 (累计 {total_ingested}/{len(sequences)})"
        )

        if batch_idx < len(batches) - 1:
            time.sleep(0.1)

    return total_ingested


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ETT数据摄入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="ETT文件名（不含路径，如 ETTm1）",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="要摄入的列名（如 OT, HUFL）。不指定则摄入全部列。",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=INPUT_LENGTH,
        help=f"历史窗口长度（默认 {INPUT_LENGTH}）",
    )
    parser.add_argument(
        "--future-len",
        type=int,
        default=DEFAULT_FUTURE_LENGTH,
        help=f"预测窗口长度（默认 {DEFAULT_FUTURE_LENGTH}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"每批摄入条数（默认 {BATCH_SIZE}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览数据，不写入Qdrant",
    )
    args = parser.parse_args()

    # 检查数据目录
    data_dir = Path(__file__).parent.parent / "ETT_data"
    if not data_dir.exists():
        print(f"错误: ETT_data 目录不存在: {data_dir}")
        sys.exit(1)

    # 选择数据集
    if args.file:
        csv_files = [data_dir / f"{args.file}.csv"]
        if not csv_files[0].exists():
            csv_files = list(data_dir.glob(f"*{args.file}*.csv"))
    else:
        csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"错误: 在 {data_dir} 中未找到CSV文件")
        sys.exit(1)

    for i, f in enumerate(csv_files):
        print(f"  [{i}] {f.name}")

    if len(csv_files) == 1:
        selected_file = csv_files[0]
    elif args.file:
        matches = [f for f in csv_files if args.file in f.name]
        if len(matches) == 1:
            selected_file = matches[0]
        else:
            print(f"\n多个文件匹配 '{args.file}'，请手动选择:")
            for i, f in enumerate(csv_files):
                print(f"  [{i}] {f.name}")
            choice = input("请输入编号: ").strip()
            selected_file = csv_files[int(choice)]
    else:
        print(f"\n请选择要摄入的数据集:")
        for i, f in enumerate(csv_files):
            print(f"  [{i}] {f.name}")
        choice = input("请输入编号 (默认 0): ").strip() or "0"
        selected_file = csv_files[int(choice)]

    print(f"\n{'='*60}")
    print(f"选中的数据集: {selected_file.name}")
    print(f"模式: {'[Dry Run - 仅预览]' if args.dry_run else '[写入模式]'}")
    print(f"{'='*60}")

    df, numeric_cols = load_ett_data(str(selected_file))

    # 选择列
    if args.column:
        if args.column not in numeric_cols:
            print(f"错误: 列 '{args.column}' 不存在，可用列: {numeric_cols}")
            sys.exit(1)
        columns_to_ingest = [args.column]
    else:
        print(f"\n可用列: {numeric_cols}")
        print("将摄入全部列。每列数据独立生成样本。")
        columns_to_ingest = numeric_cols

    # 初始化组件（仅写入模式）
    processor = None
    encoder = None
    retriever = None

    if not args.dry_run:
        import portalocker

        lock_file = Path(STORAGE_PATH) / ".lock"
        if lock_file.exists():
            try:
                with open(lock_file, "rb") as f:
                    portalocker.lock(
                        f,
                        portalocker.LockFlags.EXCLUSIVE
                        | portalocker.LockFlags.NON_BLOCKING,
                    )
                    portalocker.unlock(f)
            except portalocker.exceptions.AlreadyLocked:
                print("\n错误: 检测到 Qdrant 数据目录正被其他进程占用。")
                print("请先停止 FastAPI 服务再运行摄入脚本：")
                print("  终端中按 Ctrl+C 停止 uvicorn 服务")
                print("  摄入完成后，重新运行: uvicorn main:app --reload --port 8000")
                sys.exit(1)

        print("\n初始化组件...")
        processor = TSProcessor()
        encoder = ONNXEncoder(onnx_path=ONNX_PATH, input_length=INPUT_LENGTH)
        retriever = QdrantRetriever(storage_path=STORAGE_PATH)
        retriever.create_collection_if_not_exists(
            collection_name=COLLECTION_NAME,
            vector_size=encoder.get_embedding_dim(),
        )
        print(f"  ONNX模型: {encoder.get_embedding_dim()}维向量")
        print(f"  存储路径: {STORAGE_PATH}")

    # 摄入
    total_samples = 0
    for col in columns_to_ingest:
        print(f"\n{'─'*60}")
        print(f"摄入列: {col}")
        print(f"{'─'*60}")
        count = ingest_column(
            df=df,
            column=col,
            history_len=args.history_len,
            future_len=args.future_len,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            processor=processor,
            encoder=encoder,
            retriever=retriever,
        )
        total_samples += count

    print(f"\n{'='*60}")
    print(f"完成! 共摄入 {total_samples} 条样本 (来自 {len(columns_to_ingest)} 列)")
    print(f"数据集: {selected_file.name}")
    print(f"历史窗口: {args.history_len}, 预测窗口: {args.future_len}")
    print(f"Payload 时间特征: month, hour, is_weekend, time_of_day")
    if args.dry_run:
        print("提示: 使用 --file/--column 等参数配合执行实际写入")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
