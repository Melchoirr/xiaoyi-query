import requests
import duckdb
import pandas as pd
import numpy as np
import json
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

DB_PATH = "gdelt_master.duckdb"
KEYWORD = "Federal Reserve"
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"

# ==========================================
# 1. DATA FETCHING (Cloudflare Bypassing & Auto-Retry)
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    logging.info(f"[Polymarket] Searching for markets related to '{keyword}'...")
    session = requests.Session()
    
    # 🌟 完善的真实浏览器请求头伪装
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://polymarket.com",
        "Referer": "https://polymarket.com/"
    })
    
    try:
        res = session.get("https://gamma-api.polymarket.com/public-search", params={"q": keyword, "limit_per_type": 10, "keep_closed_markets": 1}, timeout=10).json()
    except Exception as e:
        logging.error(f"Search API failed: {e}")
        return pd.DataFrame()

    events = res.get("events", [])
    if not events: return pd.DataFrame()

    for ev in events:
        cids = json.loads(ev['markets'][0]['clobTokenIds']) if isinstance(ev['markets'][0]['clobTokenIds'], str) else ev['markets'][0]['clobTokenIds']
        token_id = cids[0]
        title = ev.get('title', 'Unknown')
        
        logging.info(f"[Polymarket] Trying Event: {title[:50]}...")
        
        all_history = []
        curr_start = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        # 10 days per chunk
        chunk_seconds = 10 * 24 * 3600 
        
        while curr_start < end_ts:
            curr_end = min(curr_start + chunk_seconds, end_ts)
            params = {"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60}
            
            # 🌟 带有自动重试的请求模块
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    r = session.get("https://clob.polymarket.com/prices-history", params=params, timeout=10)
                    if r.status_code == 200:
                        all_history.extend(r.json().get("history", []))
                    break # 成功则跳出重试循环
                except requests.exceptions.SSLError as e:
                    logging.warning(f"  [!] SSL Blocked by Cloudflare (Attempt {attempt+1}/{max_retries}). Retrying in 2s...")
                    time.sleep(2) # 遇到阻断，停顿 2 秒再试
                except Exception as e:
                    logging.error(f"  [!] Request Error: {e}")
                    break
            
            curr_start = curr_end
            time.sleep(0.5) # 🌟 增加正常的请求间隙，防止被封 IP

        if all_history:
            df = pd.DataFrame(all_history)
            df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df['price'] = pd.to_numeric(df['p'])
            grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
            df_aligned = df.drop_duplicates('datetime_utc').set_index('datetime_utc')['price'].resample('h').last().reindex(grid).ffill().bfill()
            
            logging.info(f"  └─ 🎉 Success! Fetched {len(df_aligned)} aligned hours.")
            return df_aligned.reset_index().rename(columns={'index': 'datetime_utc'})
            
    logging.error("No historical data found for this period.")
    return pd.DataFrame()

def fetch_gdelt_news(keyword, start_str, end_str):
    s_gdelt = start_str.replace("-","").replace(" ","").replace(":","")
    e_gdelt = end_str.replace("-","").replace(" ","").replace(":","")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        query = f"""
        SELECT strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
               COUNT(*) AS news_volume
        FROM gkg_news 
        WHERE CAST(DATE AS VARCHAR) >= '{s_gdelt}' AND CAST(DATE AS VARCHAR) <= '{e_gdelt}'
              AND Estimated_Title ILIKE '%{keyword}%'
        GROUP BY datetime_utc ORDER BY datetime_utc
        """
        df_news = con.execute(query).df()
        con.close()
        if not df_news.empty:
            df_news['datetime_utc'] = pd.to_datetime(df_news['datetime_utc']).dt.tz_localize('UTC')
            return df_news
    except Exception as e: logging.error(f"DuckDB Error: {e}")
    return pd.DataFrame()

# ==========================================
# 2. VOLATILITY & ATTENTION ENGINE
# ==========================================
def run_volatility_analysis():
    start_dt, end_dt = pd.to_datetime(START_DATE, utc=True), pd.to_datetime(END_DATE, utc=True)
    
    df_poly = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    df_news = fetch_gdelt_news(KEYWORD, START_DATE, END_DATE)
    
    if df_poly.empty or df_news.empty:
        logging.error("Insufficient data.")
        return

    df = pd.merge(df_poly, df_news, on='datetime_utc', how='left').fillna(0)

    # ---------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------
    # 1. Market Volatility (Absolute change in price)
    df['volatility'] = df['price'].diff().abs()
    
    # 2. News Attention Z-Score (Rolling 24h)
    # Calculate moving average and standard deviation over the last 24 hours
    df['vol_mean_24h'] = df['news_volume'].rolling(window=24, min_periods=1).mean()
    df['vol_std_24h'] = df['news_volume'].rolling(window=24, min_periods=1).std().replace(0, 1) # Prevent div by 0
    
    # Z-Score: How abnormal is the current hour's volume compared to the last 24 hours?
    df['news_zscore'] = (df['news_volume'] - df['vol_mean_24h']) / df['vol_std_24h']
    
    # Drop NaNs
    df = df.dropna()

    # Calculate overall correlation
    corr_val = df['news_zscore'].corr(df['volatility'])
    
    print("\n" + "="*50)
    print("🔥 VOLATILITY vs ATTENTION ANALYSIS 🔥")
    print("="*50)
    print(f"Correlation between News Z-Score and Market Volatility: {corr_val:.3f}")
    if corr_val > 0.15:
        print("=> Strong Signal: Major news spikes reliably trigger market repricing.")
    else:
        print("=> Weak Signal: Market seems to move independently of media hype volume.")
    print("="*50 + "\n")

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))
    
    # Chart 1: Time Series (Aligned Spikes)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title(f"Time Series: Market Volatility vs Abnormal News Spikes ({KEYWORD})", fontsize=14)
    
    ax1.plot(df['datetime_utc'], df['volatility'], color='tab:blue', linewidth=2, label='Market Volatility (abs Price Diff)')
    ax1.set_ylabel('Market Volatility', color='tab:blue', fontsize=12)
    
    ax2 = ax1.twinx()
    # Only highlight extreme spikes (Z-Score > 2)
    extreme_spikes = df[df['news_zscore'] > 2]
    ax2.bar(df['datetime_utc'], df['news_zscore'], color='tab:gray', alpha=0.3, width=0.03, label='Normal News Volume')
    ax2.bar(extreme_spikes['datetime_utc'], extreme_spikes['news_zscore'], color='tab:red', alpha=0.7, width=0.04, label='Extreme News Spikes (Z > 2)')
    ax2.axhline(2, color='red', linestyle='--', alpha=0.5) # Z=2 threshold
    ax2.set_ylabel('News Attention (Z-Score)', color='tab:red', fontsize=12)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Chart 2: Scatter Plot with Regression Line
    ax3 = plt.subplot(2, 1, 2)
    ax3.set_title(f"Scatter Plot: Does News Hype drive Market Volatility?", fontsize=14)
    
    sns.regplot(
        x=df['news_zscore'], 
        y=df['volatility'], 
        ax=ax3, 
        scatter_kws={'alpha':0.4, 'color': 'tab:gray'}, 
        line_kws={'color':'tab:red', 'linewidth': 2}
    )
    
    ax3.set_xlabel('News Attention (Z-Score: Standard Deviations above 24h Mean)', fontsize=12)
    ax3.set_ylabel('Market Volatility (Absolute Price Change)', fontsize=12)
    
    # Add text box with correlation
    ax3.text(
        0.05, 0.9, f"Pearson Correlation: {corr_val:.3f}", 
        transform=ax3.transAxes, fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig("volatility_attention.png", dpi=300)
    logging.info("Chart saved as 'volatility_attention.png'")
    plt.show()

if __name__ == "__main__":
    run_volatility_analysis()