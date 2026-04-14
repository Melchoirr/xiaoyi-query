import os
import zipfile
import re
import pandas as pd
from tqdm import tqdm

# ================= 1. 配置参数 =================
zip_dir = 'gdelt_gkg_zips'          # 你的 ZIP 文件夹路径
output_csv = 'gdelt_gkg_data.csv'   # 最终生成的 CSV 文件名
log_file = 'processed_files.txt'    # 用于记录已处理文件名的日志（断点续传）

# GDELT V2 GKG 标准列名 (原始 27 列)
gkg_columns =[
    'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
    'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
    'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
    'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
    'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
    'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
]

# 我们精选保留的列
columns_to_keep =[
    'GKGRECORDID', 'DATE', 'SourceCommonName', 'DocumentIdentifier', 
    'V2Tone'
]

# ================= 2. 辅助函数：提取标题 =================
def extract_title_from_url(url):
    try:
        if not isinstance(url, str): return None
        slug = url.strip('/').split('/')[-1]
        slug = re.sub(r'\.(html|htm|php|asp|aspx|cms|stml|amp)$', '', slug)
        title = slug.replace('-', ' ').replace('_', ' ')
        if title.replace(' ', '').isdigit() or len(title) < 8:
            return None
        return title.strip().title()
    except:
        return None

# ================= 3. 加载已处理文件列表 (断点续传) =================
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        processed_set = set(line.strip() for line in f if line.strip())
else:
    processed_set = set()

# ================= 4. 扫描文件 =================
print("🔍 正在扫描本地 ZIP 文件...")
all_files = [f for f in os.listdir(zip_dir) if f.endswith('.gkg.csv.zip')]
pending_files = [f for f in all_files if f not in processed_set]
pending_files.sort()

print(f"📦 发现总文件数: {len(all_files)}")
print(f"✅ 已处理: {len(processed_set)}")
print(f"⏳ 待处理: {len(pending_files)}\n")

if len(pending_files) == 0:
    print("🎉 所有文件都已处理完毕！")
    exit()

# ================= 5. 开始批量解析与保存 =================
# 检查 CSV 是否已存在，以决定是否写入表头
file_exists = os.path.isfile(output_csv)

with tqdm(total=len(pending_files), desc="💾 导出进度", unit="包", colour="green") as pbar:
    for file_name in pending_files:
        file_path = os.path.join(zip_dir, file_name)
        pbar.set_postfix_str(f"处理中: {file_name}")
        
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                # 获取压缩包内的文件名
                inner_file = z.namelist()[0]
                with z.open(inner_file) as f:
                    # 1. 读入 Pandas
                    df = pd.read_csv(f, sep='\t', encoding='utf-8', on_bad_lines='skip', 
                                     encoding_errors='replace', header=None, 
                                     names=gkg_columns, dtype=str, low_memory=False)
                    
                    # 2. 剔除多余列
                    df = df[columns_to_keep].copy()
                    
                    # 3. 提取标题
                    df['Estimated_Title'] = df['DocumentIdentifier'].apply(extract_title_from_url)
                    
                    # 调整顺序
                    df = df[['GKGRECORDID', 'DATE', 'SourceCommonName', 'DocumentIdentifier', 
                             'Estimated_Title', 'V2Tone']]
                    
                    # 4. 追加保存到 CSV
                    # header=not file_exists 表示只有在文件第一次创建时才写表头
                    df.to_csv(output_csv, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                    
                    # 更新文件存在状态
                    file_exists = True
                    
                    # 5. 记录已处理文件
                    with open(log_file, 'a', encoding='utf-8') as log:
                        log.write(file_name + '\n')
                        
        except Exception as e:
            print(f"\n❌ 文件 {file_name} 解析失败: {e}")
            
        pbar.update(1)

# ================= 6. 总结 =================
print("\n" + "="*40)
print(f"🎉 处理完成！")
print(f"📄 结果保存至: {os.path.abspath(output_csv)}")
print(f"📝 进度记录至: {os.path.abspath(log_file)}")
print("="*40)