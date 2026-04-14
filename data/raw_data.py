import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 1. 配置参数 =================
start_time = '2026-02-25 00:00:00'
end_time   = '2026-03-25 23:59:59'
download_dir = 'gdelt_gkg_zips'

# 🌟 核心提速参数：同时下载几个文件？
# 建议设为 5 到 10 之间。太高会被 GDELT 封锁 IP，10 是一个很安全且极速的甜点值。
MAX_WORKERS = 10 

os.makedirs(download_dir, exist_ok=True)
time_range = pd.date_range(start=start_time, end=end_time, freq='15min')
base_url = "http://data.gdeltproject.org/gdeltv2/"

print(f"🚀 启动 GDELT 多线程极速下载器 | 并发数: {MAX_WORKERS}")
print(f"📦 预计总文件数: {len(time_range)} 个\n")

# ================= 2. 定义单个下载任务 =================
def download_single_file(task_info):
    file_name, download_url, file_path = task_info
    
    # 防重复检查
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
        return ('skipped', file_name)
        
    try:
        # 优化点：把 chunk_size 调大到 64KB (65536)，减少 I/O 写入次数，提升速度
        response = requests.get(download_url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk: 
                        f.write(chunk)
            return ('success', file_name)
        elif response.status_code == 404:
            return ('404', file_name)
        else:
            return ('error', file_name)
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path) # 清理残缺文件
        return ('exception', file_name)

# ================= 3. 构造任务池并执行 =================
tasks =[]
for dt in time_range:
    timestamp_str = dt.strftime('%Y%m%d%H%M00')
    file_name = f"{timestamp_str}.gkg.csv.zip"
    tasks.append((
        file_name, 
        base_url + file_name, 
        os.path.join(download_dir, file_name)
    ))

# 统计数据
stats = {'success': 0, 'skipped': 0, '404': 0, 'error': 0, 'exception': 0}

# 使用多线程池 + tqdm 进度条
with tqdm(total=len(tasks), desc="📦 总进度", unit="包", colour="green") as pbar:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将任务提交给线程池
        futures = {executor.submit(download_single_file, task): task for task in tasks}
        
        # 只要有一个线程完成，就更新进度条
        for future in as_completed(futures):
            status, completed_file_name = future.result()
            stats[status] += 1
            
            # 动态更新进度条后缀
            if status == 'success':
                pbar.set_postfix_str(f"✅ 下载完毕: {completed_file_name}")
            elif status == 'skipped':
                pbar.set_postfix_str(f"⏭️ 已跳过: {completed_file_name}")
                
            pbar.update(1)

print("\n" + "="*40)
print(f"🎉 并发下载任务全部完成！")
print(f"📥 新下载: {stats['success']} 个")
print(f"⏭️ 已跳过: {stats['skipped']} 个 (本地已有)")
print(f"⚠️ 缺失/错误: {stats['404'] + stats['error'] + stats['exception']} 个")
print("="*40)