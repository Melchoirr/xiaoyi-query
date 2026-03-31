import requests
import json
import pandas as pd
import time
import os
import logging
from datetime import datetime
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区域
# ==========================================

KEYWORDS =["trump", "fed", "bitcoin"]

START_DATE = "2025-01-01 00:00:00"
END_DATE = "2026-03-30 00:00:00"
GRANULARITY = "1h"
FORWARD_FILL_PRICES = True
MAX_EVENTS_PER_KEYWORD = 50  # 建议调大，否则旧市场会被新市场挤掉

# 🌟 是否搜索已结束(结算)的历史市场？
# True: 包含已结束市场 (如果你想找很久以前的事件，必须设为 True)
# False: 只搜当前活跃市场
KEEP_CLOSED_MARKETS = True

# 🌟 数据写入模式
# True: 追加模式 (Append) - 新爬取的数据会接着写在 CSV 后面，适合分批爬取
# False: 覆盖模式 (Overwrite) - 每次运行都会清空旧文件，重新生成 CSV
APPEND_TO_CSV = False

# 生命周期过滤
# ==========================================
# 🛡️ 事件生命周期过滤参数 (双保险：API级 + 本地级)
# 格式：ISO 8601 (例如 "2025-01-01T00:00:00Z")，不填留空 "" 即可
# ==========================================
EVENT_START_TIME_MIN = ""
EVENT_START_TIME_MAX = "2025-03-30T00:00:00Z"
EVENT_CLOSED_TIME_MIN = ""
EVENT_CLOSED_TIME_MAX = ""

OUTPUT_CSV = "polymarket_dataset_final.csv"
LOG_FILE = "polymarket_scraper.log"

# ==========================================
# 📝 日志配置
# ==========================================
logger = logging.getLogger("PolyScraper")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w')  # 日志文件默认覆盖，保持干净
fh.setFormatter(formatter)
logger.addHandler(fh)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def is_event_valid_locally(ev):
    try:
        if EVENT_START_TIME_MIN:
            t_min = pd.to_datetime(EVENT_START_TIME_MIN, utc=True)
            ev_start = pd.to_datetime(ev.get('startDate', '1970-01-01T00:00:00Z'), utc=True)
            if ev_start < t_min: return False
        if EVENT_START_TIME_MAX:
            t_max = pd.to_datetime(EVENT_START_TIME_MAX, utc=True)
            ev_start = pd.to_datetime(ev.get('startDate', '2100-01-01T00:00:00Z'), utc=True)
            if ev_start > t_max: return False
        return True
    except: return True

def fetch_events_by_keyword(keyword):
    url = "https://gamma-api.polymarket.com/public-search"
    params = {
        "q": keyword, 
        "limit_per_type": MAX_EVENTS_PER_KEYWORD,
        "keep_closed_markets": 1 if KEEP_CLOSED_MARKETS else 0
    }
    
    try:
        resp = requests.get(url, params=params, headers=HEADERS)
        if resp.status_code == 200:
            events = resp.json().get("events",[])
            valid_events = [ev for ev in events if is_event_valid_locally(ev)]
            logger.info(f"搜索 '{keyword}': API返回 {len(events)} 个，经过本地历史过滤保留 {len(valid_events)} 个")
            return valid_events
        return[]
    except Exception as e:
        logger.error(f"搜索接口报错: {e}")
        return[]

def fetch_price_history_chunked(token_id, start_ts, end_ts, fidelity_minutes):
    url = "https://clob.polymarket.com/prices-history"
    all_history =[]
    
    if fidelity_minutes <= 1: safe_days = 1
    elif fidelity_minutes <= 10: safe_days = 3
    elif fidelity_minutes <= 60: safe_days = 10
    else: safe_days = 90
        
    chunk_size_seconds = safe_days * 24 * 60 * 60
    current_start = start_ts
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_size_seconds, end_ts)
        params = {
            "market": token_id,
            "startTs": current_start,
            "endTs": current_end,
            "fidelity": fidelity_minutes
        }
        
        try:
            resp = requests.get(url, params=params, headers=HEADERS)
            if resp.status_code == 200:
                data = resp.json().get("history",[])
                all_history.extend(data)
        except Exception as e:
            logger.error(f"  [分片异常] {e}")
            
        current_start = current_end
        time.sleep(0.3)
        
    return all_history

def main():
    print(f"🚀 Polymarket 数据集构建引擎 (高级配置版)")
    try:
        start_dt = pd.to_datetime(START_DATE, utc=True)
        end_dt = pd.to_datetime(END_DATE, utc=True)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
    except Exception as e:
        print(f"❌ 时间解析失败: {e}")
        return

    fid_map = {'1m': (1, '1min'), '10m': (10, '10min'), '1h': (60, '1h'), '1d': (1440, '1D')}
    fidelity_minutes, pd_freq = fid_map.get(GRANULARITY, (60, '1h'))
    standard_time_grid = pd.date_range(start=start_dt, end=end_dt, freq=pd_freq)
    
    print("🔎 正在检索符合条件的市场...")
    tasks =[]
    
    for keyword in KEYWORDS:
        events = fetch_events_by_keyword(keyword)
        for ev in events:
            ev_title = ev.get('title', 'Untitled')
            for m in ev.get("markets",[]):
                question = m.get('question', 'Unknown')
                clob_ids_raw = m.get('clobTokenIds')
                outcomes_raw = m.get('outcomes')
                
                if not clob_ids_raw or not outcomes_raw: continue
                    
                try:
                    tokens = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                except: continue
                
                for idx, outcome_name in enumerate(outcomes):
                    if idx >= len(tokens): break
                    tasks.append({
                        'keyword': keyword,
                        'event_title': ev_title,
                        'question': question,
                        'outcome_name': outcome_name,
                        'token_id': tokens[idx]
                    })
                    
    if not tasks:
        print("❌ 未发现任何符合条件的市场，请检查时间过滤参数。")
        return

    all_dataframes =[]
    
    for task in tqdm(tasks, desc="📈 爬取进度", unit="项", ncols=100):
        token_id = task['token_id']
        logger.info(f"\n>>> 开始处理: {task['question'][:30]} -> {task['outcome_name']}")
        
        raw_data = fetch_price_history_chunked(token_id, start_ts, end_ts, fidelity_minutes)
        
        if raw_data:
            df_raw = pd.DataFrame(raw_data)
            df_raw['datetime_utc'] = pd.to_datetime(df_raw['t'], unit='s', utc=True)
            df_raw = df_raw[(df_raw['datetime_utc'] >= start_dt) & (df_raw['datetime_utc'] <= end_dt)]
            df_raw = df_raw.drop_duplicates(subset=['datetime_utc'], keep='last')
            df_raw.set_index('datetime_utc', inplace=True)
            df_raw.sort_index(inplace=True)
            fetched_points = len(df_raw)
        else:
            fetched_points = 0
        
        # 🌟 核心：吸附对齐算法
        if fetched_points > 0:
            df_resampled = df_raw.resample(pd_freq).last()
            df_aligned = df_resampled.reindex(standard_time_grid)
        else:
            df_aligned = pd.DataFrame(index=standard_time_grid, columns=['p'])

        # 价格空值前向填充
        if FORWARD_FILL_PRICES and fetched_points > 0:
            df_aligned['p'] = df_aligned['p'].ffill()
            
        # 写入元数据标签
        df_aligned['keyword'] = task['keyword']
        df_aligned['event_title'] = task['event_title']
        df_aligned['market_question'] = task['question']
        df_aligned['outcome'] = task['outcome_name']
        df_aligned['token_id'] = token_id
        
        df_aligned.reset_index(inplace=True)
        df_aligned.rename(columns={'index': 'datetime_utc', 'p': 'price'}, inplace=True)
        if 't' in df_aligned.columns: df_aligned.drop(columns=['t'], inplace=True)
            
        all_dataframes.append(df_aligned)
        time.sleep(0.3)

    print("\n🔄 正在合并数据集...")
    if all_dataframes:
        final_dataset = pd.concat(all_dataframes, ignore_index=True)
        
        # 重新排列列顺序：Event 和 Market 放到最前面
        cols =[
            'event_title', 'market_question', 'outcome', 
            'datetime_utc', 'price', 'keyword', 'token_id'
        ]
        final_dataset = final_dataset[cols]
        
        # ==========================================
        # 🌟 智能写入逻辑 (追加 or 覆盖)
        # ==========================================
        file_exists = os.path.isfile(OUTPUT_CSV)
        
        if APPEND_TO_CSV and file_exists:
            # 追加写入：mode='a', 并且不写 header，以免表头重复
            final_dataset.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"🎉 任务完成！{len(final_dataset)} 行新数据已【追加】至 {OUTPUT_CSV}")
        else:
            # 覆盖写入 / 新建文件：mode='w'
            final_dataset.to_csv(OUTPUT_CSV, mode='w', header=True, index=False, encoding='utf-8-sig')
            if APPEND_TO_CSV and not file_exists:
                print(f"🎉 任务完成！由于文件原先不存在，已【新建】文件 {OUTPUT_CSV} (写入 {len(final_dataset)} 行)")
            else:
                print(f"🎉 任务完成！数据已【覆盖】保存至 {OUTPUT_CSV} (写入 {len(final_dataset)} 行)")
    else:
        print("❌ 未抓取到有效数据。")

if __name__ == "__main__":
    main()