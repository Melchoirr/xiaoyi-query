import requests
import duckdb
import pandas as pd
import json
import logging
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

DB_PATH = "gdelt_master.duckdb"    # Your DuckDB path
KEYWORD = "OpenAI"                 # Search Keyword
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"
OUTPUT_PNG = "alignment_result.png"

# ==========================================
# 1. POLYMARKET ENGINE (with Chunking & Headers)
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    logging.info(f"[Polymarket] Searching for markets related to '{keyword}'...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })

    # Search for events
    search_url = "https://gamma-api.polymarket.com/public-search"
    res = session.get(search_url, params={"q": keyword, "limit_per_type": 10, "keep_closed_markets": 1}).json()
    
    events = res.get("events", [])
    if not events:
        logging.error("No events found on Polymarket.")
        return pd.DataFrame()

    history_url = "https://clob.polymarket.com/prices-history"
    
    for ev in events:
        title = ev.get('title', 'Unknown')
        market = ev['markets'][0]
        cids = json.loads(market['clobTokenIds']) if isinstance(market['clobTokenIds'], str) else market['clobTokenIds']
        token_id = cids[0]

        logging.info(f"[Polymarket] Trying Event: {title[:50]}...")
        
        # Chunked Fetching (10 days per chunk for 1h fidelity)
        all_history = []
        curr_start = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        chunk_sec = 10 * 24 * 60 * 60 

        while curr_start < end_ts:
            curr_end = min(curr_start + chunk_sec, end_ts)
            params = {"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60}
            try:
                r = session.get(history_url, params=params)
                if r.status_code == 200:
                    all_history.extend(r.json().get("history", []))
            except Exception as e:
                logging.error(f"Chunk error: {e}")
            curr_start = curr_end
            time.sleep(0.2)

        if all_history:
            df = pd.DataFrame(all_history)
            df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df['price'] = pd.to_numeric(df['p'])
            df = df.drop_duplicates('datetime_utc').set_index('datetime_utc').sort_index()
            
            # Reindex to a perfect hourly grid
            grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
            df_aligned = df['price'].resample('h').last().reindex(grid).ffill().bfill()
            
            final_df = df_aligned.reset_index().rename(columns={'index': 'datetime_utc'})
            logging.info(f"[Polymarket] Success! Retrieved {len(final_df)} data points.")
            return final_df

    return pd.DataFrame()

# ==========================================
# 2. DUCKDB GDELT ENGINE (Correct Date Format)
# ==========================================
def fetch_gdelt_news(keyword, start_str, end_str):
    logging.info(f"[DuckDB] Querying GDELT news for '{keyword}'...")
    # Convert "2026-02-25 00:00:00" -> "20260225000000"
    s_gdelt = start_str.replace("-","").replace(" ","").replace(":","")
    e_gdelt = end_str.replace("-","").replace(" ","").replace(":","")

    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        query = f"""
        SELECT 
            strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
            COUNT(*) AS news_volume,
            AVG(CAST(split_part(V2Tone, ',', 1) AS FLOAT)) AS avg_tone
        FROM gkg_news 
        WHERE 
            CAST(DATE AS VARCHAR) >= '{s_gdelt}' 
            AND CAST(DATE AS VARCHAR) <= '{e_gdelt}'
            AND Estimated_Title ILIKE '%{keyword}%'
        GROUP BY datetime_utc
        ORDER BY datetime_utc
        """
        df_news = con.execute(query).df()
        con.close()
        
        if not df_news.empty:
            df_news['datetime_utc'] = pd.to_datetime(df_news['datetime_utc']).dt.tz_localize('UTC')
            logging.info(f"[DuckDB] Found {df_news['news_volume'].sum()} articles across {len(df_news)} hours.")
            return df_news
    except Exception as e:
        logging.error(f"DuckDB Error: {e}")
    return pd.DataFrame()

# ==========================================
# 3. VISUALIZATION ENGINE (Dynamic Y-Axis)
# ==========================================
def plot_results(df):
    logging.info("[Viz] Generating charts with dynamic Y-axis...")
    
    # Calculate Dynamic Y-axis limits for Price
    p_min = df['price'].min()
    p_max = df['price'].max()
    p_range = p_max - p_min
    
    # Add a 10% margin above and below the data range
    # If the range is 0 (price is flat), use a default offset
    margin = p_range * 0.1 if p_range > 0 else 0.05
    y_lower = max(0, p_min - margin) # Don't go below 0 probability
    y_upper = min(1.0, p_max + margin) # Don't go above 1.0 probability

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Price & News Volume
    ax1.set_title(f"Alignment: Polymarket Price vs News Volume ({KEYWORD})", fontsize=14)
    ax1.plot(df['datetime_utc'], df['price'], color='tab:blue', linewidth=2, label='Market Price', zorder=3)
    ax1.set_ylabel('Probability (Price)', color='tab:blue', fontsize=12, fontweight='bold')
    
    # Applying Dynamic Limits
    ax1.set_ylim(y_lower, y_upper)
    
    ax2 = ax1.twinx()
    ax2.bar(df['datetime_utc'], df['news_volume'], color='tab:gray', alpha=0.3, width=0.02, label='News Volume', zorder=1)
    ax2.set_ylabel('Hourly News Count', color='tab:gray')
    ax2.grid(False)

    # Plot 2: Price & Sentiment
    ax3.set_title(f"Alignment: Polymarket Price vs News Sentiment ({KEYWORD})", fontsize=14)
    ax3.plot(df['datetime_utc'], df['price'], color='tab:blue', linewidth=2, label='Market Price', zorder=3)
    ax3.set_ylabel('Probability (Price)', color='tab:blue', fontsize=12, fontweight='bold')
    
    # Applying Dynamic Limits
    ax3.set_ylim(y_lower, y_upper)

    ax4 = ax3.twinx()
    ax4.plot(df['datetime_utc'], df['avg_tone'], color='tab:orange', marker='.', linestyle='--', alpha=0.6, label='Sentiment (Tone)', zorder=2)
    ax4.axhline(0, color='red', linestyle=':', alpha=0.5)
    ax4.set_ylabel('Avg Sentiment Score', color='tab:orange')
    ax4.grid(False)

    # Format Time Axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    logging.info(f"[Viz] Dynamic view saved as {OUTPUT_PNG}")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)

    # 1. Get Data
    df_poly = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    if df_poly.empty: return

    df_news = fetch_gdelt_news(KEYWORD, START_DATE, END_DATE)

    # 2. Merge
    logging.info("[Main] Merging datasets...")
    if not df_news.empty:
        merged = pd.merge(df_poly, df_news, on='datetime_utc', how='left')
    else:
        merged = df_poly.copy()
        merged['news_volume'], merged['avg_tone'] = 0, 0

    merged['news_volume'] = merged['news_volume'].fillna(0)
    merged['avg_tone'] = merged['avg_tone'].fillna(0)

    # 3. Plot
    plot_results(merged)

if __name__ == "__main__":
    main()