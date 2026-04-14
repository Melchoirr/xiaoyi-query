import os
import time
import zipfile
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 1. 配置参数 =================
start_time = '2026-01-01 00:00:00'
end_time   = '2026-01-02 23:59:59'
download_dir = 'gdelt_gkg_zips'

# 🌟 核心提速参数
MAX_WORKERS = 10 
# 🔄 自动重试次数
MAX_RETRIES = 3  

os.makedirs(download_dir, exist_ok=True)
time_range = pd.date_range(start=start_time, end=end_time, freq='15min')
base_url = "http://data.gdeltproject.org/gdeltv2/"

print(f"🚀 启动 GDELT 多线程极速下载器 | 并发数: {MAX_WORKERS} | 重试次数: {MAX_RETRIES}")
print(f"📦 预计总文件数: {len(time_range)} 个\n")

# ================= 2. 完整性检查函数 =================
def is_valid_zip(file_path):
    """检查文件是否存在且是一个完整的 zip 文件"""
    if not os.path.exists(file_path):
        return False
    # 如果文件太小（比如几十字节的报错信息），直接判为无效
    if os.path.getsize(file_path) < 1024:
        return False
        
    try:
        # 尝试读取 zip 文件，testzip() 会检查 CRC 和文件头
        with zipfile.ZipFile(file_path, 'r') as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                return False  # zip 内部有损坏
        return True
    except (zipfile.BadZipFile, Exception):
        return False # 不是有效的 zip 文件

# ================= 3. 定义单个下载任务 (含重试逻辑) =================
def download_single_file(task_info):
    file_name, download_url, file_path = task_info
    
    # 【1】防重复与完整性检查 (本地已有且完整，直接跳过)
    if is_valid_zip(file_path):
        return ('skipped', file_name)
        
    # 【2】下载与重试循环
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 清理之前的残缺文件
            if os.path.exists(file_path):
                os.remove(file_path)
                
            response = requests.get(download_url, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk: 
                            f.write(chunk)
                            
                # 【3】下载完成后立即进行完整性检查
                if is_valid_zip(file_path):
                    return ('success', file_name)
                else:
                    # 下载了但是损坏了，继续下一次重试
                    time.sleep(1) # 稍微停顿一下再重试
                    continue
                    
            elif response.status_code == 404:
                # 404 文件不存在，无需重试，直接返回
                return ('404', file_name)
            else:
                # 其他 HTTP 错误 (如 500, 502 等)，等待后重试
                time.sleep(2)
                continue
                
        except requests.exceptions.RequestException:
            # 网络异常 (超时、断开等)，等待后重试
            time.sleep(2)
            continue
            
    # 【4】如果循环结束还没 return，说明重试耗尽仍失败
    if os.path.exists(file_path):
        os.remove(file_path) # 确保不留残缺文件
    return ('failed', file_name)

# ================= 4. 构造任务池并执行 =================
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
stats = {'success': 0, 'skipped': 0, '404': 0, 'failed': 0}

# 使用多线程池 + tqdm 进度条
with tqdm(total=len(tasks), desc="📦 总进度", unit="包", colour="green") as pbar:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_file, task): task for task in tasks}
        
        for future in as_completed(futures):
            status, completed_file_name = future.result()
            stats[status] += 1
            
            # 动态更新进度条后缀
            if status == 'success':
                pbar.set_postfix_str(f"✅ 下载成功: {completed_file_name}")
            elif status == 'skipped':
                pbar.set_postfix_str(f"⏭️ 完整跳过: {completed_file_name}")
            elif status == 'failed':
                pbar.set_postfix_str(f"❌ 重试失败: {completed_file_name}")
                
            pbar.update(1)

print("\n" + "="*45)
print(f"🎉 并发下载任务全部完成！")
print(f"📥 成功下载: {stats['success']} 个")
print(f"⏭️ 本地已有(且完整): {stats['skipped']} 个")
print(f"👻 GDELT缺失(404): {stats['404']} 个")
print(f"❌ 损坏/下载失败: {stats['failed']} 个")
print("="*45)