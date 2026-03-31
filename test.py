import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def flexible_parse_ts(date_str):
    if not date_str or str(date_str).strip() == "": return None
    try:
        dt = pd.to_datetime(date_str)
        return int(dt.timestamp())
    except: return None

def get_price_history_strict(token_id, start_ts=None, end_ts=None, fidelity_minutes=60):
    url = "https://clob.polymarket.com/prices-history"
    params = {"market": token_id}
    
    if start_ts:
        params["startTs"] = start_ts
        # 即使 API 可能忽略 endTs，我们依然发送它（万一哪天它生效了呢）
        if end_ts: params["endTs"] = end_ts
        params["fidelity"] = fidelity_minutes
    else:
        params["interval"] = "max"
        
    print(f"[*] 发送 API 请求: {params}")
    
    try:
        resp = requests.get(url, params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"❌ API 错误: {resp.text}")
            return []
            
        history = resp.json().get("history", [])
        
        # --- 核心改进：本地严格过滤 ---
        if start_ts and end_ts:
            initial_count = len(history)
            # 只保留在 [start_ts, end_ts] 范围内的数据
            history = [item for item in history if start_ts <= item['t'] <= end_ts]
            filtered_count = initial_count - len(history)
            if filtered_count > 0:
                print(f"[*] API 返回了多余数据，已在本地过滤掉未来/超出范围的 {filtered_count} 条记录。")
        
        return history
    except Exception as e:
        print(f"❌ 异常: {e}")
        return []

# ... (public_search 等函数保持不变) ...

def main():
    print("=== Polymarket 历史数据精准提取工具 (带本地截断) ===")
    query = input("1. 输入关键词搜索事件: ")
    events = requests.get("https://gamma-api.polymarket.com/public-search", 
                          params={"q": query, "limit_per_type": 10}, headers=HEADERS).json().get("events", [])
    
    if not events: return

    for i, ev in enumerate(events): print(f"[{i}] {ev.get('title')}")
    ev_idx = int(input("\n选择事件: "))

    markets = events[ev_idx].get("markets", [])

    for i, m in enumerate(markets):
        print(f"[{i}] {m.get('question')}")
    sel_market = events[ev_idx].get("markets", [])[int(input("选择市场编号: "))]

    token_ids = json.loads(sel_market['clobTokenIds']) if isinstance(sel_market['clobTokenIds'], str) else sel_market['clobTokenIds']
    outcomes = json.loads(sel_market['outcomes']) if isinstance(sel_market['outcomes'], str) else sel_market['outcomes']
    
    for i, o in enumerate(outcomes): print(f"[{i}] {o}")
    target_token = token_ids[int(input("选择选项: "))]

    print("\n--- 时间范围设置 ---")
    s_date = input("开始日期 (YYYY-MM-DD): ").strip()
    e_date = input("结束日期 (YYYY-MM-DD): ").strip()
    
    start_ts = flexible_parse_ts(s_date)
    end_ts = flexible_parse_ts(e_date)
    
    fid = 60
    if start_ts:
        gran = input("粒度 (1m/10m/1h/1d): ").strip()
        fid = {'1m':1, '10m':10, '1h':60, '1d':1440}.get(gran, 60)

    # 获取并过滤数据
    raw = get_price_history_strict(target_token, start_ts, end_ts, fid)

    if not raw:
        print("未获取到该时段数据。")
        return

    df = pd.DataFrame(raw)
    df['timestamp'] = pd.to_datetime(df['t'], unit='s')
    df['price'] = df['p']

    print("\n" + "═"*50)
    print(f"📈 精准数据集报告")
    print(f" - 起始时间: {df['timestamp'].min()}")
    print(f" - 结束时间: {df['timestamp'].max()}")
    print(f" - 记录总数: {len(df)} 条")
    print("═"*50)

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['price'], color='#c0392b')
    plt.title(f"Cleaned Dataset: {sel_market['question']}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == "__main__":
    main()