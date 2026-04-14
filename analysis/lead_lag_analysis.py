import requests
import duckdb
import pandas as pd
import numpy as np
import json
import logging
import time
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

DB_PATH = "gdelt_master.duckdb"
KEYWORD = "OpenAI"
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"
MAX_LAG_HOURS = 12  # 我们想观察前后 12 个小时的错位关系

# ==========================================
# 1. DATA FETCHING (Same robust logic as before)
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0"})
    res = session.get("https://gamma-api.polymarket.com/public-search", params={"q": keyword, "limit_per_type": 10, "keep_closed_markets": 1}).json()
    events = res.get("events", [])
    if not events: return pd.DataFrame()

    for ev in events:
        cids = json.loads(ev['markets'][0]['clobTokenIds']) if isinstance(ev['markets'][0]['clobTokenIds'], str) else ev['markets'][0]['clobTokenIds']
        token_id = cids[0]
        
        all_history = []
        curr_start = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        while curr_start < end_ts:
            curr_end = min(curr_start + 10 * 24 * 3600, end_ts)
            r = session.get("https://clob.polymarket.com/prices-history", params={"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60})
            if r.status_code == 200: all_history.extend(r.json().get("history", []))
            curr_start = curr_end
            time.sleep(0.2)

        if all_history:
            df = pd.DataFrame(all_history)
            df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df['price'] = pd.to_numeric(df['p'])
            grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
            df_aligned = df.drop_duplicates('datetime_utc').set_index('datetime_utc')['price'].resample('h').last().reindex(grid).ffill().bfill()
            return df_aligned.reset_index().rename(columns={'index': 'datetime_utc'})
    return pd.DataFrame()

def fetch_gdelt_news(keyword, start_str, end_str):
    s_gdelt = start_str.replace("-","").replace(" ","").replace(":","")
    e_gdelt = end_str.replace("-","").replace(" ","").replace(":","")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        query = f"""
        SELECT strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
               COUNT(*) AS news_volume, AVG(CAST(split_part(V2Tone, ',', 1) AS FLOAT)) AS avg_tone
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
# 2. CROSS-CORRELATION ANALYSIS ENGINE
# ==========================================
def crosscorr(datax, datay, lag=0):
    """
    Calculate cross-correlation between datax and datay with a specific lag.
    If lag > 0: datax is compared against PAST datay (datay leads datax).
    If lag < 0: datax is compared against FUTURE datay (datax leads datay).
    """
    return datax.corr(datay.shift(lag))

def run_lead_lag_analysis():
    start_dt, end_dt = pd.to_datetime(START_DATE, utc=True), pd.to_datetime(END_DATE, utc=True)
    
    # 1. Fetch & Merge
    df_poly = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    df_news = fetch_gdelt_news(KEYWORD, START_DATE, END_DATE)
    
    if df_poly.empty or df_news.empty:
        logging.error("Insufficient data for analysis.")
        return

    df = pd.merge(df_poly, df_news, on='datetime_utc', how='left').fillna(0)

    # 2. Feature Engineering (CRITICAL for valid correlation)
    # Price difference (Return)
    df['price_diff'] = df['price'].diff()
    # Volatility (Absolute Return) - measures how violent the market is
    df['volatility'] = df['price_diff'].abs()
    
    # Drop first row because diff() produces NaN
    df = df.dropna()
    
    if df['price_diff'].std() == 0:
        logging.error("Price never changed during this period. Correlation is undefined.")
        return

    lags = range(-MAX_LAG_HOURS, MAX_LAG_HOURS + 1)
    
    # Analysis A: News Volume vs Market Volatility (Attention -> Volatility)
    # lag > 0 means correlating current volatility with PAST news volume.
    vol_corrs = [crosscorr(df['volatility'], df['news_volume'], lag) for lag in lags]
    
    # Analysis B: News Tone vs Price Direction (Sentiment -> Return)
    # lag > 0 means correlating current price_diff with PAST news tone.
    tone_corrs = [crosscorr(df['price_diff'], df['avg_tone'], lag) for lag in lags]

    # Print Insights
    best_vol_lag = lags[np.argmax(np.abs(vol_corrs))]
    best_tone_lag = lags[np.argmax(np.abs(tone_corrs))]
    
    print("\n" + "="*50)
    print("🧠 LEAD-LAG ANALYSIS INSIGHTS")
    print("="*50)
    print(f"🔹 Volume vs Volatility Peak Correlation: {max(np.abs(vol_corrs)):.3f} at Lag {best_vol_lag} hours.")
    if best_vol_lag > 0:
        print(f"   => NEWS LEADS: News volume spikes predict market volatility {best_vol_lag} hour(s) later.")
    elif best_vol_lag < 0:
        print(f"   => PRICE LEADS: Market volatility predicts news volume spikes {abs(best_vol_lag)} hour(s) later (Media reacts to market).")
    else:
        print(f"   => COINCIDENT: News and market volatility happen in the exact same hour.")
        
    print("\n" + "-"*50)

    # 3. Visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot A
    colors1 = ['tab:red' if x < 0 else ('tab:green' if x > 0 else 'tab:gray') for x in lags]
    ax1.bar(lags, vol_corrs, color=colors1, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Cross-Correlation: News Volume vs Market Volatility", fontsize=14)
    ax1.set_ylabel("Correlation Coefficient (Pearson)", fontsize=12)
    ax1.text(-MAX_LAG_HOURS*0.9, max(vol_corrs)*0.8, "⬅ Price Predicts News\n(Market leads Media)", color='tab:red', fontsize=10)
    ax1.text(MAX_LAG_HOURS*0.1, max(vol_corrs)*0.8, "News Predicts Price ➡\n(Media leads Market)", color='tab:green', fontsize=10)
    
    # Plot B
    colors2 = ['tab:red' if x < 0 else ('tab:green' if x > 0 else 'tab:gray') for x in lags]
    ax2.bar(lags, tone_corrs, color=colors2, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Cross-Correlation: News Sentiment (Tone) vs Price Return", fontsize=14)
    ax2.set_xlabel("Lag (Hours)", fontsize=12)
    ax2.set_ylabel("Correlation Coefficient (Pearson)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("lead_lag_analysis.png", dpi=300)
    logging.info("Chart saved as 'lead_lag_analysis.png'")
    plt.show()

if __name__ == "__main__":
    run_lead_lag_analysis()