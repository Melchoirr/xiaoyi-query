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
KEYWORD = "China"
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"

# 🌟 向量匹配相似度阈值 (0 到 1)
# 0.35 属于中等严格。如果过滤后新闻数量为 0，可以适当调低到 0.25
SIMILARITY_THRESHOLD = 0.35 

# ==========================================
# 1. POLYMARKET 获取 (提取精确的事件问题)
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    logging.info(f"[Polymarket] Searching for '{keyword}'...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0",
        "Origin": "https://polymarket.com", "Referer": "https://polymarket.com/"
    })
    
    res = session.get("https://gamma-api.polymarket.com/public-search", params={"q": keyword, "limit_per_type": 10, "keep_closed_markets": 1}, timeout=10).json()
    events = res.get("events", [])
    if not events: return pd.DataFrame(), ""

    for ev in events:
        market = ev['markets'][0]
        cids = json.loads(market['clobTokenIds']) if isinstance(market['clobTokenIds'], str) else market['clobTokenIds']
        token_id = cids[0]
        
        # 🌟 关键：提取 Polymarket 的精确问题，用于后续的 AI 向量比对！
        event_question = ev.get('title', '') + " " + market.get('question', '')
        logging.info(f"[Polymarket] Target Event: {event_question}")
        
        all_history = []
        curr_start = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        while curr_start < end_ts:
            curr_end = min(curr_start + 10 * 24 * 3600, end_ts)
            for attempt in range(3):
                try:
                    r = session.get("https://clob.polymarket.com/prices-history", params={"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60}, timeout=10)
                    if r.status_code == 200: all_history.extend(r.json().get("history", []))
                    break
                except requests.exceptions.SSLError: time.sleep(2)
                except Exception: break
            curr_start = curr_end
            time.sleep(0.5)

        if all_history:
            df = pd.DataFrame(all_history)
            df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df['price'] = pd.to_numeric(df['p'])
            grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
            df_aligned = df.drop_duplicates('datetime_utc').set_index('datetime_utc')['price'].resample('h').last().reindex(grid).ffill().bfill()
            
            return df_aligned.reset_index().rename(columns={'index': 'datetime_utc'}), event_question
            
    return pd.DataFrame(), ""

# ==========================================
# 2. DUCKDB 粗排 (获取所有相关的新闻标题)
# ==========================================
def fetch_raw_gdelt_titles(keyword, start_str, end_str):
    logging.info(f"[DuckDB] Fetching raw candidate news for '{keyword}'...")
    s_gdelt = start_str.replace("-","").replace(" ","").replace(":","")
    e_gdelt = end_str.replace("-","").replace(" ","").replace(":","")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        # 注意：这里不再直接 Count，而是把标题全拉出来供 AI 审阅
        query = f"""
        SELECT 
            strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
            Estimated_Title
        FROM gkg_news 
        WHERE CAST(DATE AS VARCHAR) >= '{s_gdelt}' AND CAST(DATE AS VARCHAR) <= '{e_gdelt}'
              AND Estimated_Title ILIKE '%{keyword}%'
        """
        df_news = con.execute(query).df()
        con.close()
        
        if not df_news.empty:
            df_news['datetime_utc'] = pd.to_datetime(df_news['datetime_utc']).dt.tz_localize('UTC')
            logging.info(f"[DuckDB] Retrieved {len(df_news)} candidate articles. Passing to AI for semantic filtering...")
            return df_news
    except Exception as e: logging.error(f"DuckDB Error: {e}")
    return pd.DataFrame()

# ==========================================
# 3. 核心：轻量级向量匹配 (AI 精排)
# ==========================================
def semantic_filter_news(df_news, query_text):
    logging.info("[AI] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    # 这个模型极小 (~80MB)，跑在笔记本上速度飞快
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info("[AI] Embedding market question and news titles. This may take a few seconds...")
    # 将目标问题变成向量
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    # 将所有新闻标题变成向量
    titles = df_news['Estimated_Title'].fillna("").tolist()
    title_embeddings = model.encode(titles, convert_to_tensor=True)
    
    # 计算余弦相似度
    cosine_scores = util.cos_sim(query_embedding, title_embeddings)[0].cpu().numpy()
    df_news['similarity'] = cosine_scores
    
    # 根据阈值过滤
    df_filtered = df_news[df_news['similarity'] >= SIMILARITY_THRESHOLD].copy()
    df_garbage = df_news[df_news['similarity'] < SIMILARITY_THRESHOLD]
    
    print("\n" + "═"*60)
    print("🤖 AI 过滤直击 (展示 AI 的工作成果)")
    print("═"*60)
    print(f"🎯 目标事件: {query_text}\n")
    
    print("✅ 保留的高质量新闻 (相似度极高):")
    for title in df_filtered.sort_values('similarity', ascending=False)['Estimated_Title'].head(3):
        print(f"  [+] {title}")
        
    print("\n🗑️ 被当做噪音丢弃的新闻 (只含关键词但语义无关):")
    for title in df_garbage.sample(min(3, len(df_garbage)))['Estimated_Title']:
        print(f"  [-] {title}")
    print("═"*60 + "\n")
    
    # 按小时汇总那些被保留下来的“纯净新闻”
    df_aggregated = df_filtered.groupby('datetime_utc').size().reset_index(name='news_volume')
    logging.info(f"[AI] Filtered down from {len(df_news)} to {len(df_filtered)} highly relevant articles.")
    return df_aggregated

# ==========================================
# 4. 分析与绘图主程序
# ==========================================
def run_vector_analysis():
    start_dt, end_dt = pd.to_datetime(START_DATE, utc=True), pd.to_datetime(END_DATE, utc=True)
    
    # 1. 拿数据
    df_poly, event_question = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    if df_poly.empty: return
    
    df_raw_news = fetch_raw_gdelt_titles(KEYWORD, START_DATE, END_DATE)
    if df_raw_news.empty: return
    
    # 2. 向量过滤
    df_clean_news = semantic_filter_news(df_raw_news, event_question)
    if df_clean_news.empty:
        logging.warning("No news survived the AI filter. Try lowering SIMILARITY_THRESHOLD.")
        return
        
    # 3. 合并特征
    df = pd.merge(df_poly, df_clean_news, on='datetime_utc', how='left').fillna(0)
    
    df['volatility'] = df['price'].diff().abs()
    df['vol_mean_24h'] = df['news_volume'].rolling(window=24, min_periods=1).mean()
    df['vol_std_24h'] = df['news_volume'].rolling(window=24, min_periods=1).std().replace(0, 1)
    df['news_zscore'] = (df['news_volume'] - df['vol_mean_24h']) / df['vol_std_24h']
    
    df = df.dropna()
    corr_val = df['news_zscore'].corr(df['volatility'])
    
    print(f"📈 [结果] 清洗后的 新闻异动(Z-Score) 与 市场波动率 的相关系数: {corr_val:.3f}")

    # 4. 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图 1
    ax1.set_title(f"Time Series: Market Volatility vs NLP-Filtered News Spikes", fontsize=14)
    ax1.plot(df['datetime_utc'], df['volatility'], color='tab:blue', linewidth=2, label='Market Volatility')
    ax1.set_ylabel('Market Volatility (abs Price Diff)', color='tab:blue')
    
    ax2 = ax1.twinx()
    extreme_spikes = df[df['news_zscore'] > 2]
    ax2.bar(df['datetime_utc'], df['news_zscore'], color='tab:gray', alpha=0.3, width=0.03, label='Normal Z-Score')
    ax2.bar(extreme_spikes['datetime_utc'], extreme_spikes['news_zscore'], color='tab:red', alpha=0.7, width=0.04, label='Spikes (Z>2)')
    ax2.set_ylabel('Filtered News Z-Score', color='tab:red')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # 图 2
    ax3.set_title(f"Scatter Plot: Does Highly Relevant News drive Volatility?", fontsize=14)
    sns.regplot(x=df['news_zscore'], y=df['volatility'], ax=ax3, scatter_kws={'alpha':0.4, 'color':'tab:gray'}, line_kws={'color':'tab:red', 'linewidth': 2})
    ax3.set_xlabel('Semantic Filtered News Attention (Z-Score)', fontsize=12)
    ax3.set_ylabel('Market Volatility', fontsize=12)
    ax3.text(0.05, 0.9, f"Correlation: {corr_val:.3f}", transform=ax3.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("vector_volatility.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_vector_analysis()