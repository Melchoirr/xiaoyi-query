"""Assess dataset sufficiency for price prediction modeling.

Focus: Can we predict Δprice(t+1h, t+3h, t+6h, t+12h, t+24h) using:
- Historical prices (autoregressive baseline)
- News volume at time t
- News volume change / acceleration
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

RESULT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("result/by_date/2026-05-11/kg_gdelt_alignment")


def build_features(df, event_name):
    """Build modeling features from merged price+news data."""
    df = df.copy()
    df = df.sort_values("datetime_utc")

    # Price features
    df["price"] = df["price"].interpolate().fillna(method="ffill").fillna(method="bfill")
    df["dprice_1h"] = df["price"].diff(1)
    df["dprice_3h"] = df["price"].diff(3)
    df["dprice_6h"] = df["price"].diff(6)
    df["dprice_12h"] = df["price"].diff(12)
    df["dprice_24h"] = df["price"].diff(24)

    # Rolling price stats
    df["price_ma_6h"] = df["price"].rolling(6).mean()
    df["price_ma_24h"] = df["price"].rolling(24).mean()
    df["price_std_6h"] = df["price"].rolling(6).std()
    df["price_std_24h"] = df["price"].rolling(24).std()

    # News volume features
    df["news_volume"] = df["news_volume"].fillna(0)
    df["news_ma_6h"] = df["news_volume"].rolling(6).mean()
    df["news_ma_24h"] = df["news_volume"].rolling(24).mean()
    df["news_delta_1h"] = df["news_volume"].diff(1)
    df["news_delta_6h"] = df["news_volume"].diff(6)
    df["news_zscore_24h"] = (df["news_volume"] - df["news_ma_24h"]) / df["news_volume"].rolling(24).std().replace(0, 1)

    # Surge indicator: news > 2x 24h moving average
    df["news_surge"] = (df["news_volume"] > 2 * df["news_ma_24h"]).astype(int)

    return df.dropna()


def evaluate_predictive_power(df, event_name):
    """Compare baseline (AR) vs news-augmented predictive power for Δprice."""
    horizons = [1, 3, 6, 12, 24]
    results = {"event": event_name, "n_samples": len(df)}

    for h in horizons:
        target_col = f"dprice_{h}h"
        if target_col not in df.columns:
            continue

        # Shift target BACKWARD so we predict FUTURE changes from CURRENT features
        y = df[target_col].shift(-h).dropna()
        if len(y) < 10:
            continue

        # Align features to same index
        X = df.loc[y.index]

        # Baseline: AR(1) - just use current dprice_1h
        ar_pred = X["dprice_1h"]
        ar_r, ar_p = stats.pearsonr(ar_pred, y)
        ar_mae = np.mean(np.abs(y - ar_pred))
        ar_rmse = np.sqrt(np.mean((y - ar_pred)**2))

        # News-only: just use news_zscore
        news_pred = X["news_zscore_24h"].fillna(0)
        news_r, news_p = stats.pearsonr(news_pred, y)

        # News volume raw
        vol_r, vol_p = stats.pearsonr(X["news_volume"], y)

        # News surge indicator
        surge = X["news_surge"]
        surge_mean = y[surge == 1].mean() if surge.sum() > 0 else np.nan
        non_surge_mean = y[surge == 0].mean()
        surge_effect = surge_mean - non_surge_mean if not np.isnan(surge_mean) else np.nan

        results[f"h{h}_ar_r"] = ar_r
        results[f"h{h}_ar_p"] = ar_p
        results[f"h{h}_news_z_r"] = news_r
        results[f"h{h}_news_z_p"] = news_p
        results[f"h{h}_vol_r"] = vol_r
        results[f"h{h}_vol_p"] = vol_p
        results[f"h{h}_surge_effect"] = surge_effect

    return results


def assess_dataset_quality(csv_files):
    """Overall dataset quality assessment."""
    total_hours = 0
    total_news = 0
    zero_news_hours = 0
    events_info = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        df = df.sort_values("datetime_utc")

        n = len(df)
        vol = df["news_volume"].fillna(0)
        zero_h = (vol == 0).sum()
        price_valid = df["price"].notna().sum()

        events_info.append({
            "name": csv_path.stem.split("_", 1)[1].replace("_merged_series", "").replace("-", " ")[:50],
            "hours": n,
            "price_ok": price_valid,
            "mean_vol": vol.mean(),
            "median_vol": vol.median(),
            "zero_pct": zero_h / n * 100,
            "max_vol": vol.max(),
            "vol_std": vol.std(),
        })
        total_hours += n
        total_news += vol.sum()
        zero_news_hours += zero_h

    return events_info, {
        "total_hours": total_hours,
        "total_news": total_news,
        "zero_news_pct": zero_news_hours / total_hours * 100,
        "mean_news_per_hour": total_news / total_hours,
        "n_events": len(csv_files),
    }


def main():
    csv_files = sorted(RESULT_DIR.glob("event_*_merged_series.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULT_DIR}")
        return

    # === PART 1: Dataset Quality ===
    print("=" * 90)
    print("DATASET QUALITY ASSESSMENT")
    print("=" * 90)
    events_info, overall = assess_dataset_quality(csv_files)

    print(f"{'Event':<45s} {'Hours':>6s} {'MeanVol':>8s} {'MedVol':>7s} {'MaxVol':>7s} {'Zero%':>6s}")
    print("-" * 90)
    for e in events_info:
        print(f"{e['name']:<45s} {e['hours']:>6d} {e['mean_vol']:>8.1f} {e['median_vol']:>7.1f} {e['max_vol']:>7.0f} {e['zero_pct']:>5.1f}%")

    print("-" * 90)
    print(f"  Overall: {overall['total_hours']:,} hours, {overall['total_news']:,} articles")
    print(f"  Zero-news hours: {overall['zero_news_pct']:.1f}%")
    print(f"  Mean news/hour: {overall['mean_news_per_hour']:.1f}")
    print(f"  Events: {overall['n_events']}")

    # === PART 2: Predictive Power Evaluation ===
    print("\n" + "=" * 120)
    print("PREDICTIVE POWER: News volume vs AR(1) baseline for Δprice")
    print("=" * 120)

    header = f"{'Event':<40s} {'Horiz':>6s} {'AR(1)_r':>8s} {'AR_p':>6s} {'NewsZ_r':>8s} {'News_p':>6s} {'Vol_r':>8s} {'Vol_p':>6s} {'Surge Δ':>8s}"
    print(header)
    print("-" * 120)

    all_pred = []
    for csv_path in csv_files:
        event_id = csv_path.stem.split("_")[1]
        event_name = csv_path.stem.replace(f"event_{event_id}_", "").replace("_merged_series", "").replace("-", " ")[:40]
        df = pd.read_csv(csv_path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        df = df.sort_values("datetime_utc")
        df = df.dropna(subset=["price"])

        if len(df) < 48:
            continue

        features = build_features(df, event_name)
        pred = evaluate_predictive_power(features, event_name)
        if pred:
            all_pred.append(pred)
            for h in [1, 3, 6, 12, 24]:
                ar_r = pred.get(f"h{h}_ar_r", np.nan)
                ar_p = pred.get(f"h{h}_ar_p", np.nan)
                nz_r = pred.get(f"h{h}_news_z_r", np.nan)
                nz_p = pred.get(f"h{h}_news_z_p", np.nan)
                vol_r = pred.get(f"h{h}_vol_r", np.nan)
                vol_p = pred.get(f"h{h}_vol_p", np.nan)
                surge = pred.get(f"h{h}_surge_effect", np.nan)

                ar_sig = "*" if not np.isnan(ar_p) and ar_p < 0.05 else ""
                nz_sig = "*" if not np.isnan(nz_p) and nz_p < 0.05 else ""
                vol_sig = "*" if not np.isnan(vol_p) and vol_p < 0.05 else ""

                line = f"{event_name:<40s} {f'{h}h':>6s} {ar_r:>7.3f}{ar_sig} {ar_p:>5.3f} {nz_r:>7.3f}{nz_sig} {nz_p:>5.3f} {vol_r:>7.3f}{vol_sig} {vol_p:>5.3f}"
                if not np.isnan(surge):
                    line += f" {surge:>+7.4f}"
                print(line)
            print()

    # === PART 3: Verdict ===
    print("=" * 90)
    print("DATASET SUFFICIENCY VERDICT")
    print("=" * 90)

    # Count events where news adds value over AR
    news_wins = 0
    for p in all_pred:
        for h in [1, 3, 6, 12, 24]:
            ar_r = abs(p.get(f"h{h}_ar_r", 0))
            nz_r = abs(p.get(f"h{h}_news_z_r", 0))
            if not np.isnan(nz_r) and nz_r > ar_r:
                news_wins += 1

    total_comparisons = len(all_pred) * 5
    print(f"\n  News z-score beats AR(1) in {news_wins}/{total_comparisons} horizon-event pairs")

    # Data sparsity check
    sparse_events = [e for e in events_info if e["zero_pct"] > 50]
    rich_events = [e for e in events_info if e["mean_vol"] > 50]

    print(f"\n  Rich-data events (mean > 50 news/h): {len(rich_events)}")
    for e in rich_events:
        print(f"    - {e['name']}: {e['mean_vol']:.0f}/h, {e['zero_pct']:.0f}% zero")
    print(f"\n  Sparse-data events (> 50% zero-news hours): {len(sparse_events)}")
    for e in sparse_events:
        print(f"    - {e['name']}: {e['mean_vol']:.0f}/h, {e['zero_pct']:.0f}% zero")

    # Recommendations
    print("\n  --- Recommendations ---")
    print()
    print("  当前数据集的问题：")
    print("  1. 新闻匹配基于简单的 ILIKE 关键词，缺乏事件相关性过滤")
    print("     → 'Iran' 匹配到了所有提到伊朗的新闻，不论是否与特定事件相关")
    print("  2. 事件间的新闻存在高度重叠（多个事件共享 'Iran'、'United States' 标签）")
    print("     → 不同事件实际上在分析同一批新闻的不同侧面")
    print("  3. 缺乏事件专属的信号区分度")
    print()
    print("  改进方向：")
    print("  a) 在关键词匹配基础上，加入事件标题的语义相似度过滤")
    print("     （用 embedding 计算标题与新闻标题的 cosine similarity）")
    print("  b) 加入事件特定的细粒度实体（如 'nuclear deal' vs 仅 'iran'）")
    print("  c) 将事件作为独立时间序列，建模时加入 event_id 作为 categorical feature")
    print("  d) 对新闻去重，同一 URL 不重复计入不同事件")


if __name__ == "__main__":
    main()
