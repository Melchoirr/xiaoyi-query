import requests
import duckdb
import pandas as pd
import numpy as np
import json
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# ==========================================
# ⚙️ 配置区域
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

DB_PATH = "gdelt_master.duckdb"
KEYWORD = "ChatGPT"
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"
SIMILARITY_THRESHOLD = 0.35 

# ==========================================
# 1 & 2 & 3. 数据获取与 AI 过滤 (逻辑保持不变)
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    # 🌟 建议：运行脚本时将 KEYWORD 设为更简短的 "Claude" 
    logging.info(f"[Polymarket] Searching for markets related to '{keyword}'...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json", "Origin": "https://polymarket.com", "Referer": "https://polymarket.com/"
    })

    search_url = "https://gamma-api.polymarket.com/public-search"
    res = None
    # 增加 limit 到 20，扩大搜索范围
    try:
        resp = session.get(search_url, params={"q": keyword, "limit_per_type": 20, "keep_closed_markets": 1}, timeout=30)
        if resp.status_code == 200: res = resp.json()
    except Exception as e:
        logging.error(f"Search API Error: {e}")
        return pd.DataFrame(), ""

    events = res.get("events", [])
    if not events:
        logging.error(f"❌ 搜索结果为空！请尝试缩短关键词（例如只搜 'Claude'）。")
        return pd.DataFrame(), ""

    history_url = "https://clob.polymarket.com/prices-history"
    
    # 🌟 遍历搜索到的所有 Event
    for ev in events:
        event_title = ev.get('title', 'Untitled Event')
        markets = ev.get('markets', [])
        
        # 🌟 遍历该 Event 下的所有具体 Market（比如不同日期的 Claude 预测）
        for m in markets:
            question = m.get('question', '')
            # 只有包含关键词或者是我们要找的 April 30 才继续
            # 或者我们可以简单地全都试一遍
            cids_raw = m.get('clobTokenIds')
            if not cids_raw: continue
            
            cids = json.loads(cids_raw) if isinstance(cids_raw, str) else cids_raw
            token_id = cids[0]
            full_title = f"{event_title}: {question}"
            
            logging.info(f"🔎 正在检查市场: {full_title[:70]}...")

            all_history = []
            curr_start = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            chunk_seconds = 10 * 24 * 3600 

            while curr_start < end_ts:
                curr_end = min(curr_start + chunk_seconds, end_ts)
                params = {"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60}
                try:
                    r = session.get(history_url, params=params, timeout=30)
                    if r.status_code == 200:
                        data = r.json().get("history", [])
                        all_history.extend(data)
                except: break
                curr_start = curr_end
                time.sleep(0.3)

            if all_history:
                logging.info(f"  ✅ 命中！在目标时间段内找到 {len(all_history)} 条价格记录。")
                df = pd.DataFrame(all_history)
                df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
                df['price'] = pd.to_numeric(df['p'])
                grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
                df_aligned = df.drop_duplicates('datetime_utc').set_index('datetime_utc')['price'].resample('h').last().reindex(grid).ffill().bfill()
                return df_aligned.reset_index().rename(columns={'index': 'datetime_utc'}), full_title
            else:
                logging.warning(f"  ⚠️ 跳过：该市场在 {start_dt.date()} ~ {end_dt.date()} 之间没有交易。")

    logging.error("❌ 遍历了所有搜索结果，均未在指定时间段内找到交易数据。")
    return pd.DataFrame(), ""

def fetch_raw_gdelt_titles(keyword, start_str, end_str):
    s_gdelt = start_str.replace("-","").replace(" ","").replace(":","")
    e_gdelt = end_str.replace("-","").replace(" ","").replace(":","")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        query = f"""
        SELECT strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc, Estimated_Title
        FROM gkg_news 
        WHERE CAST(DATE AS VARCHAR) >= '{s_gdelt}' AND CAST(DATE AS VARCHAR) <= '{e_gdelt}'
              AND Estimated_Title ILIKE '%{keyword}%'
        """
        df_news = con.execute(query).df()
        con.close()
        if not df_news.empty:
            df_news['datetime_utc'] = pd.to_datetime(df_news['datetime_utc']).dt.tz_localize('UTC')
            return df_news
    except Exception as e: logging.error(f"DuckDB Error: {e}")
    return pd.DataFrame()

def semantic_filter_news(df_news, query_text):
    logging.info("[AI] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    titles = df_news['Estimated_Title'].fillna("").tolist()
    title_embeddings = model.encode(titles, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(query_embedding, title_embeddings)[0].cpu().numpy()
    df_news['similarity'] = cosine_scores
    
    df_filtered = df_news[df_news['similarity'] >= SIMILARITY_THRESHOLD].copy()
    
    print("\n" + "═"*60)
    print("🤖 AI 过滤示例")
    print(f"🎯 目标事件: {query_text}\n")
    print("✅ 保留的高质量新闻:")
    for title in df_filtered.sort_values('similarity', ascending=False)['Estimated_Title'].head(3):
        print(f"  [+] {title}")
    print("═"*60 + "\n")
    
    df_aggregated = df_filtered.groupby('datetime_utc').size().reset_index(name='news_volume')
    return df_aggregated

# ==========================================
# 4. 最直观的散点回归分析
# ==========================================
def run_direct_analysis():
    start_dt, end_dt = pd.to_datetime(START_DATE, utc=True), pd.to_datetime(END_DATE, utc=True)
    
    df_poly, event_question = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    if df_poly.empty: return
    
    df_raw_news = fetch_raw_gdelt_titles(KEYWORD, START_DATE, END_DATE)
    if df_raw_news.empty: return
    
    df_clean_news = semantic_filter_news(df_raw_news, event_question)
    if df_clean_news.empty:
        logging.warning("No news survived the AI filter. Try lowering SIMILARITY_THRESHOLD.")
        return
        
    df = pd.merge(df_poly, df_clean_news, on='datetime_utc', how='left').fillna(0)
    
    # 🌟 直接计算波动率 (绝对差值)
    df['volatility'] = df['price'].diff().abs()
    df = df.dropna()
    
    # 计算最直接的相关系数：发文量 vs 波动率
    corr_val = df['news_volume'].corr(df['volatility'])
    print(f"📈 [结果] AI过滤后的 发文量 与 市场波动率 的相关系数: {corr_val:.3f}")

    # 绘图部分
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图 1：时间线对比
    ax1.set_title(f"Time Series: Market Volatility vs Filtered News Volume", fontsize=14)
    ax1.plot(df['datetime_utc'], df['volatility'], color='tab:blue', linewidth=2, label='Market Volatility (Price Diff)')
    ax1.set_ylabel('Market Volatility', color='tab:blue', fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.bar(df['datetime_utc'], df['news_volume'], color='tab:gray', alpha=0.5, width=0.03, label='Filtered News Count')
    ax2.set_ylabel('Filtered News Count', color='tab:gray', fontsize=12)
    ax2.grid(False)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # 图 2：散点回归图 (发文篇数 -> 波动率)
    ax3.set_title(f"Scatter Plot: Does News Volume drive Volatility?", fontsize=14)
    sns.regplot(
        x=df['news_volume'], 
        y=df['volatility'], 
        ax=ax3, 
        scatter_kws={'alpha':0.6, 'color':'tab:gray'}, 
        line_kws={'color':'tab:red', 'linewidth': 2}
    )
    ax3.set_xlabel('Filtered News Count (Number of highly relevant articles per hour)', fontsize=12)
    ax3.set_ylabel('Market Volatility (Absolute Price Change)', fontsize=12)
    
    ax3.text(
        0.05, 0.9, f"Pearson Correlation: {corr_val:.3f}", 
        transform=ax3.transAxes, fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig("direct_volume_volatility.png", dpi=300)
    logging.info("Chart saved as 'direct_volume_volatility.png'")
    plt.show()

if __name__ == "__main__":
    run_direct_analysis()