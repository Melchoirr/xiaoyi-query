import requests
import json
import pandas as pd
import time
import logging
from datetime import timedelta
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚙️ 配置区域
# ==========================================
KEYWORDS =["bitcoin", "crypto", "OpenAI", "AI", 
        "Claude", "Gemini", "Bard", "LLaMA", "Falcon",
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Nvidia"]

# 🌟 强制要求：2026 年内有效跨度至少 90 天（约 3 个月）
MIN_HISTORICAL_DAYS = 90
TARGET_YEAR = 2026

GRANULARITY = "1m"
MAX_WORKERS = 3             
OUTPUT_CSV = "polymarket_real_longterm_dataset.csv"
LOG_FILE = "polymarket_spider.log"  
CHUNK_DAYS = 7                      

# ==========================================
# 📝 日志配置
# ==========================================
logger = logging.getLogger("PolySpider")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): 
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})

# ==========================================
# 🛠️ 核心分片抓取逻辑
# ==========================================

def fetch_history_in_chunks(token_id, start_dt, end_dt):
    url = "https://clob.polymarket.com/prices-history"
    all_raw_data =[]
    
    now_dt = pd.Timestamp.now(tz='UTC')
    actual_end_dt = min(end_dt, now_dt)
    
    current_start = start_dt
    logger.info(f"[Token {token_id[-6:]}] 📡 开始抓取 | 实际跨度: {start_dt.date()} -> {actual_end_dt.date()}")
    
    while current_start < actual_end_dt:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), actual_end_dt)
        fidelity_map = {
            "1m": 1,
            "10m": 10,
            "1h": 60,
            "1d": 1440
        }
        params = {
            "market": token_id,
            "startTs": int(current_start.timestamp()),
            "endTs": int(current_end.timestamp()),
            "fidelity": fidelity_map.get(GRANULARITY, 1)
        }
        
        try:
            r = session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json().get("history",[])
                all_raw_data.extend(data)
                logger.info(f"[Token {token_id[-6:]}]   ✅ 分片 {current_start.date()} ~ {current_end.date()} | HTTP 200 | 获得 {len(data)} 个点")
            elif r.status_code == 429:
                time.sleep(3)
                continue
        except Exception as e:
            logger.error(f"[Token {token_id[-6:]}]   ❌ 网络异常: {e}")
            
        current_start = current_end
        time.sleep(0.2)
        
    return pd.DataFrame(all_raw_data) if all_raw_data else None

# ==========================================
# 🔄 任务处理逻辑
# ==========================================

def process_token_task(task):
    token_id = task['token_id']
    title = task['event_title'][:15]
    logger.info(f"\n▶️ 开始处理任务: [{title}...] | Outcome: {task['outcome']}")
    
    df_raw = fetch_history_in_chunks(token_id, task['start_dt'], task['end_dt'])
    
    if df_raw is None or df_raw.empty:
        logger.error(f"[Token {token_id[-6:]}] 🚫 数据集为空，没有交易记录。")
        return None

    try:
        t_col = 't' if 't' in df_raw.columns else ('timestamp' if 'timestamp' in df_raw.columns else None)
        p_col = 'p' if 'p' in df_raw.columns else ('price' if 'price' in df_raw.columns else None)
        
        if not t_col or not p_col: return None

        logger.info(f"[Token {token_id[-6:]}] 🔄 正在清洗对齐数据，原始点数: {len(df_raw)}")
        
        is_ms = df_raw[t_col].max() > 1e11
        df_raw['dt'] = pd.to_datetime(df_raw[t_col], unit='ms' if is_ms else 's', utc=True)
        df_raw[p_col] = pd.to_numeric(df_raw[p_col], errors='coerce') 
        
        df_raw = df_raw.dropna(subset=[p_col]) 
        df_raw = df_raw.drop_duplicates('dt').set_index('dt').sort_index()
        
        actual_end_dt = min(task['end_dt'], pd.Timestamp.now(tz='UTC'))
        
        freq_map = {
            '1m': 'min',
            '10m': '10min',
            '1h': 'h',
            '1d': 'D'
        }
        freq_str = freq_map.get(GRANULARITY, 'min')

        if GRANULARITY == '1d':
            start_grid = task['start_dt'].replace(hour=0, minute=0, second=0, microsecond=0)
            end_grid = actual_end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif GRANULARITY == '1h':
            start_grid = task['start_dt'].replace(minute=0, second=0, microsecond=0)
            end_grid = actual_end_dt.replace(minute=0, second=0, microsecond=0)
        else:
            start_grid = task['start_dt'].replace(second=0, microsecond=0)
            end_grid = actual_end_dt.replace(second=0, microsecond=0)
        
        grid = pd.date_range(start=start_grid, end=end_grid, freq=freq_str, tz='UTC')
        
        df_aligned = df_raw[p_col].resample(freq_str).last().reindex(grid)
        df_aligned = df_aligned.ffill().bfill()
        
        if df_aligned.isna().all():
            logger.error(f"[Token {token_id[-6:]}] 🚫 依然全 NaN！请检查 API 数据是否异常。")
            return None

        logger.info(f"[Token {token_id[-6:]}] 🎉 处理成功！生成有效数据 {len(df_aligned)} 行。")
        
        df_final = df_aligned.reset_index().rename(columns={'index': 'datetime_utc', p_col: 'price'})
        for col in['event_title', 'question', 'outcome', 'token_id', 'keyword', 'historical_days']:
            df_final[col] = task.get(col)
            
        return df_final
    except Exception as e:
        logger.error(f"[Token {token_id[-6:]}] 🚫 处理时发生异常: {e}")
        logger.debug(traceback.format_exc())
        return None


def get_chunk_days_for_granularity(granularity):
    if granularity == '1m':
        return 1
    if granularity == '10m':
        return 3
    if granularity == '1h':
        return 7
    return CHUNK_DAYS


def get_target_year_window(year):
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00", tz='UTC')
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59", tz='UTC')
    return year_start, year_end


def calc_overlap_days(start_dt, end_dt, window_start, window_end):
    overlap_start = max(start_dt, window_start)
    overlap_end = min(end_dt, window_end)
    if overlap_end <= overlap_start:
        return 0
    return (overlap_end - overlap_start).total_seconds() / 86400

# ==========================================
# 🏁 主程序
# ==========================================

def main():
    logger.info(f"🚀 Polymarket 爬虫启动 | 真实历史数据门槛: >= {MIN_HISTORICAL_DAYS} 天 | 写入日志文件: {LOG_FILE}")
    
    tasks =[]
    seen_tokens = set()
    now_dt = pd.Timestamp.now(tz='UTC')
    target_year_start, target_year_end = get_target_year_window(TARGET_YEAR)

    global CHUNK_DAYS
    CHUNK_DAYS = get_chunk_days_for_granularity(GRANULARITY)
    
    for kw in KEYWORDS:
        logger.info(f"🔎 正在搜索关键词: {kw}")
        try:
            url = "https://gamma-api.polymarket.com/public-search"
            # 为了能在严格门槛下搜到数据，建议把深度拉满
            res = session.get(url, params={"q": kw, "limit_per_type": 500, "keep_closed_markets": 1}).json()
            events = res.get("events",[])
            
            passed_count = 0
            for ev in events:
                s_str = ev.get('startDate') or ev.get('createdAt')
                e_str = ev.get('endDate') or ev.get('closingDate')
                if not s_str: continue
                
                start_dt = pd.to_datetime(s_str, utc=True)
                
                # 名义结束时间
                nominal_end_dt = pd.to_datetime(e_str, utc=True) if e_str else now_dt

                # 只保留 2026 年内发生且有效跨度 >= 90 天的事件
                actual_end_dt = min(nominal_end_dt, now_dt)
                historical_days = calc_overlap_days(start_dt, actual_end_dt, target_year_start, target_year_end)

                if historical_days < MIN_HISTORICAL_DAYS: 
                    continue

                overlap_start = max(start_dt, target_year_start)
                overlap_end = min(actual_end_dt, target_year_end)

                for m in ev.get("markets",[]):
                    cids = json.loads(m['clobTokenIds']) if isinstance(m['clobTokenIds'], str) else m['clobTokenIds']
                    outs = json.loads(m['outcomes']) if isinstance(m['outcomes'], str) else m['outcomes']
                    for idx, out in enumerate(outs):
                        tid = cids[idx]
                        if tid not in seen_tokens:
                            tasks.append({
                                'token_id': tid, 'keyword': kw, 'event_title': ev['title'],
                                'question': m['question'], 'outcome': out,
                                'start_dt': overlap_start, 'end_dt': overlap_end,
                                'historical_days': historical_days # 记录它的实际历史天数
                            })
                            seen_tokens.add(tid)
                            passed_count += 1
            
            logger.info(f"   ✅ 关键词 '{kw}' 筛出 {passed_count} 个任务 (这些任务已经产生了超过 {MIN_HISTORICAL_DAYS} 天的 K线)。")
        except Exception as e:
            logger.error(f"搜索 {kw} 报错: {e}")

    if not tasks:
        logger.error(f"❌ 未能找到任务，可能是该关键词下没有存活这么久的数据。")
        return

    logger.info(f"📊 开始并发抓取，共 {len(tasks)} 个任务...")

    all_results =[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_token_task, t): t for t in tasks}
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                all_results.append(res)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols =['event_title', 'question', 'outcome', 'datetime_utc', 'price', 'historical_days', 'keyword', 'token_id']
        final_df[cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 大功告成！完美长线数据集保存至 {OUTPUT_CSV} | 总行数: {len(final_df)}")
    else:
        logger.error("❌ 爬虫结束。未拿到有效数据。")

if __name__ == "__main__":
    main()