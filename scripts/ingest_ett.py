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


def load_ett_data(csv_path: str) -> pd.DataFrame:
    """加载ETT CSV数据"""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    numeric_cols = [c for c in df.columns if c != "date"]
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  可用列: {numeric_cols}")
    return df, numeric_cols


def sliding_window_sequences(
    series: np.ndarray,
    history_len: int,
    future_len: int
) -> list:
    """
    滑动窗口切分时序数据

    窗口结构:
        history_x: [t, t+1, ..., t+history_len-1]  长度 history_len
        future_y:  [t+history_len, ..., t+history_len+future_len-1]  长度 future_len

    Returns:
        List of (history_x, future_y, start_idx) tuples
    """
    total_len = history_len + future_len
    sequences = []
    for i in range(len(series) - total_len + 1):
        history = series[i : i + history_len]
        future = series[i + history_len : i + total_len]
        sequences.append((history, future, i))
    return sequences


def create_ingestion_batches(sequences: list, batch_size: int) -> list:
    """将序列列表拆分为批次"""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i : i + batch_size])
    return batches


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
):
    """摄入单列数据"""
    series = df[column].values.astype(np.float64)
    print(f"\n  列名: {column}")
    print(f"  序列长度: {len(series)}")

    # 处理缺失值
    nan_count = np.isnan(series).sum()
    if nan_count > 0:
        print(f"  警告: 检测到 {nan_count} 个缺失值，将跳过含NaN的窗口")
        # 用插值填补
        series = pd.Series(series).interpolate().fillna(method="bfill").fillna(method="ffill").values

    sequences = sliding_window_sequences(series, history_len, future_len)
    print(f"  滑动窗口: history={history_len}, future={future_len}")
    print(f"  可生成样本数: {len(sequences)}")

    if len(sequences) == 0:
        print(f"  跳过: 数据不足以生成样本 (需要 {history_len + future_len} 个点)")
        return 0

    if dry_run:
        print(f"  [Dry Run] 前3个样本预览:")
        for hx, fy, idx in sequences[:3]:
            print(f"    窗口[{idx}]: history_x范围=[{hx.min():.2f}, {hx.max():.2f}], "
                  f"future_y范围=[{fy.min():.2f}, {fy.max():.2f}]")
        return len(sequences)

    batches = create_ingestion_batches(sequences, batch_size)
    total_ingested = 0

    for batch_idx, batch in enumerate(batches):
        vectors = []
        payloads = []

        for history_x, future_y, _ in batch:
            normalized, mu, sigma = processor.normalize(history_x.tolist())
            embedding = encoder.encode(normalized)

            payloads.append({
                "source": column,
                "future_y": future_y.tolist(),
                "mu": float(mu),
                "sigma": float(sigma),
                "history_x": history_x.tolist(),
            })
            vectors.append(embedding)

        vector_ids = retriever.ingest_batch(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            payloads=payloads,
        )

        total_ingested += len(batch)
        progress = batch_idx + 1
        print(f"    批次 {progress}/{len(batches)}: "
              f"+{len(batch)} 条 (累计 {total_ingested}/{len(sequences)})")

        if batch_idx < len(batches) - 1:
            time.sleep(0.1)

    return total_ingested


def main():
    parser = argparse.ArgumentParser(
        description="ETT数据摄入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="ETT文件名（不含路径，如 ETTm1）"
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
        csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"错误: 在 {data_dir} 中未找到CSV文件")
        sys.exit(1)

    # 打印文件列表供选择
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

    # 初始化组件
    if not args.dry_run:
        import portalocker
        lock_file = Path(STORAGE_PATH) / ".lock"
        if lock_file.exists():
            try:
                with open(lock_file, "rb") as f:
                    portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
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
            vector_size=encoder.get_embedding_dim()
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
            processor=processor if not args.dry_run else None,
            encoder=encoder if not args.dry_run else None,
            retriever=retriever if not args.dry_run else None,
        )
        total_samples += count

    print(f"\n{'='*60}")
    print(f"完成! 共摄入 {total_samples} 条样本 (来自 {len(columns_to_ingest)} 列)")
    print(f"数据集: {selected_file.name}")
    print(f"历史窗口: {args.history_len}, 预测窗口: {args.future_len}")
    if args.dry_run:
        print("提示: 使用 --file/--column 等参数配合执行实际写入")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
