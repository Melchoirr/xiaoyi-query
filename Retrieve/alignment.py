import re
import time
import json
import logging
from urllib.parse import urlparse
import requests
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ==========================================
# ⚙️ 配置中心 (CONFIGURATION)
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# 数据源与基础参数
DB_PATH = "data/gdelt_master.duckdb"    # GDELT DuckDB 数据库路径
KEYWORD = "OpenAI"                      # 搜索主题（Polymarket 关键词 & 新闻检索语句）
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"
OUTPUT_PNG = "semantic_alignment_result.png"

# 语义检索高级参数 (Hybrid Retrieval Config)
MAX_CANDIDATES = 200000                 # SQL 粗排最大召回数
LEXICAL_TOPN = 8000                     # TF-IDF 保留候选数
MODEL_NAME = "BAAI/bge-large-en-v1.5"   # 语义向量模型
EMBEDDING_BATCH_SIZE = 128
HYBRID_ALPHA = 0.75                     # 语义相似度权重 (0.75表示以语义为主，词汇为辅)
MIN_EMBEDDING_SIM = 0.45                # 最低相似度阈值
FALLBACK_KEEP_RATIO = 0.08              # 如果都低于阈值，强制保留的比例

STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "will", "would", "could", "should", "by", "with", "from", "at", "as", "about", "into", "after",
    "before", "between", "than", "that", "this", "these", "those", "it", "its", "be", "been", "being",
}

# ==========================================
# 1. POLYMARKET 数据引擎
# ==========================================
def fetch_polymarket_data(keyword, start_dt, end_dt):
    logging.info(f"[Polymarket] Searching for markets related to '{keyword}'...")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })

    search_url = "https://gamma-api.polymarket.com/public-search"
    res = session.get(search_url, params={"q": keyword, "limit_per_type": 10, "keep_closed_markets": 1}).json()
    
    events = res.get("events",[])
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
        
        all_history =[]
        curr_start = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        chunk_sec = 10 * 24 * 60 * 60 

        while curr_start < end_ts:
            curr_end = min(curr_start + chunk_sec, end_ts)
            params = {"market": token_id, "startTs": curr_start, "endTs": curr_end, "fidelity": 60}
            try:
                r = session.get(history_url, params=params)
                if r.status_code == 200:
                    all_history.extend(r.json().get("history",[]))
            except Exception as e:
                logging.error(f"Chunk error: {e}")
            curr_start = curr_end
            time.sleep(0.2)

        if all_history:
            df = pd.DataFrame(all_history)
            df['datetime_utc'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df['price'] = pd.to_numeric(df['p'])
            df = df.drop_duplicates('datetime_utc').set_index('datetime_utc').sort_index()
            
            grid = pd.date_range(start=start_dt, end=end_dt, freq='h', tz='UTC')
            df_aligned = df['price'].resample('h').last().reindex(grid).ffill().bfill()
            
            final_df = df_aligned.reset_index().rename(columns={'index': 'datetime_utc'})
            logging.info(f"[Polymarket] Success! Retrieved {len(final_df)} hourly data points.")
            return final_df

    return pd.DataFrame()

# ==========================================
# 2. GDELT 新闻语义检索与清洗引擎
# ==========================================
def extract_title_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip(): return ""
    try:
        path = urlparse(url).path
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"\.(html|htm|php|asp|aspx|cms|amp|stml)$", "", slug, flags=re.IGNORECASE)
        clean = re.sub(r"[-_]+", " ", slug).strip()
        return clean if len(clean) >= 6 else ""
    except:
        return ""

def build_recall_terms(search_query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']+", search_query.lower())
    return list(set([tok for tok in tokens if len(tok) >= 3 and tok not in STOPWORDS]))

def query_candidates(start_date, end_date, search_query):
    start_key = start_date.replace("-", "").replace(" ", "").replace(":", "")
    end_key = end_date.replace("-", "").replace(" ", "").replace(":", "")

    where_parts =[
        f"CAST(DATE AS VARCHAR) >= '{start_key}'",
        f"CAST(DATE AS VARCHAR) <= '{end_key}'",
        "DocumentIdentifier IS NOT NULL",
    ]

    recall_terms = build_recall_terms(search_query)
    if recall_terms:
        likes =[f"(Estimated_Title ILIKE '%{kw.replace('''"''', "''")}%' OR DocumentIdentifier ILIKE '%{kw.replace('''"''', "''")}%')" for kw in recall_terms]
        where_parts.append("(" + " OR ".join(likes) + ")")

    sql = f"""
        SELECT
            strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
            DocumentIdentifier AS url,
            Estimated_Title AS title,
            CAST(split_part(V2Tone, ',', 1) AS FLOAT) AS tone
        FROM gkg_news
        WHERE {' AND '.join(where_parts)}
        ORDER BY CAST(DATE AS VARCHAR) DESC
        LIMIT {MAX_CANDIDATES}
    """
    
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(sql).df()
    con.close()

    if df.empty: return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize("UTC")
    df["title"] = df["title"].fillna("")
    url_title = df["url"].fillna("").map(extract_title_from_url)
    df.loc[df["title"].str.len() < 6, "title"] = url_title[df["title"].str.len() < 6]
    df = df[df["title"].fillna("").str.len() >= 6].copy()
    
    logging.info(f"[DuckDB] Coarse Recall completed. Found {len(df)} candidates.")
    return df

def rank_and_aggregate_news(df, search_query):
    if df.empty: return pd.DataFrame()

    logging.info("[Retrieval] Stage-2: TF-IDF Lexical Prefilter...")
    titles = df["title"].tolist()
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    mat = vec.fit_transform([search_query] + titles)
    lex_scores = cosine_similarity(mat[0:1], mat[1:]).reshape(-1)

    lexical_keep_n = int(max(100, LEXICAL_TOPN))
    if len(df) > lexical_keep_n:
        lex_idx = pd.Series(lex_scores).nlargest(lexical_keep_n).index
        work = df.iloc[lex_idx].copy().reset_index(drop=True)
        work_lex = lex_scores[lex_idx]
    else:
        work = df.copy().reset_index(drop=True)
        work_lex = lex_scores

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is missing! Please pip install it.")

    logging.info(f"[Retrieval] Stage-3: Semantic Embedding Rerank (Model: {MODEL_NAME})...")
    query_for_embed = f"Represent this sentence for searching relevant passages: {search_query}" if "bge" in MODEL_NAME.lower() else search_query
    
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query_for_embed], normalize_embeddings=True)
    t_emb = model.encode(work["title"].tolist(), normalize_embeddings=True, batch_size=EMBEDDING_BATCH_SIZE)
    emb_scores = (t_emb @ q_emb[0]).astype(float)

    work["hybrid_score"] = HYBRID_ALPHA * emb_scores + (1 - HYBRID_ALPHA) * work_lex
    keep = work[emb_scores >= MIN_EMBEDDING_SIM].copy()
    
    if keep.empty:
        keep_n = max(1, int(len(work) * FALLBACK_KEEP_RATIO))
        keep = work.nlargest(keep_n, "hybrid_score").copy()

    logging.info(f"[Retrieval] Filtered to {len(keep)} highly relevant articles. Aggregating to hourly bins...")
    
    # 聚合到小时级别，对接 Polymarket 时间线
    agg_df = keep.groupby('datetime_utc').agg(
        news_volume=('title', 'count'),
        avg_tone=('tone', 'mean')
    ).reset_index()
    
    return agg_df

# ==========================================
# 3. 数据融合与可视化引擎
# ==========================================
def plot_results(df):
    logging.info("[Viz] Generating aligned charts...")
    
    p_min, p_max = df['price'].min(), df['price'].max()
    margin = (p_max - p_min) * 0.1 if p_max > p_min else 0.05
    y_lower = max(0, p_min - margin)
    y_upper = min(1.0, p_max + margin)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 上半部分：价格 vs 新闻精确声量
    ax1.set_title(f"Semantic Alignment: Polymarket Price vs Refined News Volume ({KEYWORD})", fontsize=14)
    ax1.plot(df['datetime_utc'], df['price'], color='tab:blue', linewidth=2, label='Market Price', zorder=3)
    ax1.set_ylabel('Probability (Price)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.set_ylim(y_lower, y_upper)
    
    ax2 = ax1.twinx()
    ax2.bar(df['datetime_utc'], df['news_volume'], color='tab:gray', alpha=0.4, width=0.03, label='Filtered News Volume', zorder=1)
    ax2.set_ylabel('High-Relevance Hourly News', color='tab:gray')
    ax2.grid(False)

    # 下半部分：价格 vs 新闻精确情感
    ax3.set_title(f"Semantic Alignment: Polymarket Price vs Refined News Sentiment ({KEYWORD})", fontsize=14)
    ax3.plot(df['datetime_utc'], df['price'], color='tab:blue', linewidth=2, label='Market Price', zorder=3)
    ax3.set_ylabel('Probability (Price)', color='tab:blue', fontsize=12, fontweight='bold')
    ax3.set_ylim(y_lower, y_upper)

    ax4 = ax3.twinx()
    ax4.plot(df['datetime_utc'], df['avg_tone'], color='tab:orange', marker='.', linestyle='--', alpha=0.7, label='Sentiment (Tone)', zorder=2)
    ax4.axhline(0, color='red', linestyle=':', alpha=0.5)
    ax4.set_ylabel('Avg Sentiment Score', color='tab:orange')
    ax4.grid(False)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    logging.info(f"[Viz] Saved highly aligned view as {OUTPUT_PNG}")
    plt.show()

# ==========================================
# 主流程逻辑
# ==========================================
def main():
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)

    # 1. 获取 Polymarket 价格数据
    df_poly = fetch_polymarket_data(KEYWORD, start_dt, end_dt)
    if df_poly.empty: return

    # 2. 粗召回 -> 语义排序过滤 -> 按小时聚合 的完整新闻链路
    df_candidates = query_candidates(START_DATE, END_DATE, KEYWORD)
    df_news_hourly = rank_and_aggregate_news(df_candidates, KEYWORD)

    # 3. 时序对齐融合
    logging.info("[Main] Merging Semantic GDELT logic with Polymarket Timeline...")
    if not df_news_hourly.empty:
        merged = pd.merge(df_poly, df_news_hourly, on='datetime_utc', how='left')
    else:
        merged = df_poly.copy()
        merged['news_volume'], merged['avg_tone'] = 0, 0

    # 缺失填充：该小时没有高相关度新闻记作0声量，情感也归0处理
    merged['news_volume'] = merged['news_volume'].fillna(0)
    merged['avg_tone'] = merged['avg_tone'].fillna(0)

    # 4. 绘图
    plot_results(merged)

if __name__ == "__main__":
    main()