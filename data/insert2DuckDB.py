import os
import zipfile
import io
import re
import duckdb
import pandas as pd
from tqdm import tqdm

# ================= 1. 配置参数 =================
zip_dir = 'gdelt_gkg_zips'          # 你的 ZIP 文件夹路径
db_path = 'gdelt_master.duckdb'     # 生成的 DuckDB 数据库文件名

# GDELT V2 GKG 标准列名 (原始 27 列)
gkg_columns =[
    'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
    'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
    'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
    'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
    'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
    'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
]

# 我们精选保留的“有用列”（大幅节省硬盘空间，提升查询速度）
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

# ================= 3. 初始化 DuckDB 数据库 =================
print("🗄️ 正在连接 DuckDB 数据库...")
con = duckdb.connect(db_path)

# 创建存储新闻的核心表
con.execute("""
CREATE TABLE IF NOT EXISTS gkg_news (
    GKGRECORDID VARCHAR,
    DATE VARCHAR,
    SourceCommonName VARCHAR,
    DocumentIdentifier VARCHAR,
    Estimated_Title VARCHAR,   -- 🌟 我们自己生成的标题列
    V2Tone VARCHAR,
);
""")

# 创建“处理记录表”，用于断点续传防重复
con.execute("""
CREATE TABLE IF NOT EXISTS processed_files (
    filename VARCHAR PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# ================= 4. 扫描文件并比对进度 =================
print("🔍 正在扫描本地 ZIP 文件...")
all_files =[f for f in os.listdir(zip_dir) if f.endswith('.gkg.csv.zip')]

# 从数据库中获取已经处理过的文件列表
processed_df = con.execute("SELECT filename FROM processed_files").df()
processed_set = set(processed_df['filename'].tolist())

# 计算还剩下哪些文件需要处理
pending_files =[f for f in all_files if f not in processed_set]
pending_files.sort() # 按时间顺序排个序

print(f"📦 发现总文件数: {len(all_files)}")
print(f"✅ 已处理: {len(processed_set)}")
print(f"⏳ 待处理: {len(pending_files)}\n")

if len(pending_files) == 0:
    print("🎉 所有文件都已成功入库，无需重复执行！")
    exit()

# ================= 5. 开始批量解析与入库 =================
# 使用 tqdm 显示进度条
with tqdm(total=len(pending_files), desc="💾 入库进度", unit="包", colour="blue") as pbar:
    for file_name in pending_files:
        file_path = os.path.join(zip_dir, file_name)
        pbar.set_postfix_str(f"处理中: {file_name}")
        
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                with z.open(z.namelist()[0]) as f:
                    # 1. 读入 Pandas (全按字符串读取防止类型推断报错)
                    df = pd.read_csv(f, sep='\t', encoding='utf-8', on_bad_lines='skip', encoding_errors='replace',
                                     header=None, names=gkg_columns, dtype=str, low_memory=False)
                    
                    # 2. 剔除多余列，只保留精选列
                    df = df[columns_to_keep].copy()
                    
                    # 3. 施加魔法：提取标题
                    df['Estimated_Title'] = df['DocumentIdentifier'].apply(extract_title_from_url)
                    
                    # 调整列的顺序，把标题放在 URL 后面，看着舒服
                    df = df[['GKGRECORDID', 'DATE', 'SourceCommonName', 'DocumentIdentifier', 
                             'Estimated_Title', 'V2Tone']]
                    
                    # 4. 极速灌入 DuckDB
                    # 直接执行 SQL 插入 Pandas DataFrame (这是 DuckDB 的神仙操作)
                    con.execute("INSERT INTO gkg_news SELECT * FROM df")
                    
                    # 5. 标记该文件为已处理
                    con.execute("INSERT INTO processed_files (filename) VALUES (?)", (file_name,))
                    
        except Exception as e:
            pbar.set_postfix_str(f"❌ 报错跳过: {file_name}")
            print(f"\n文件 {file_name} 解析失败: {e}")
            
        pbar.update(1)

# ================= 6. 验证与总结 =================
total_records = con.execute("SELECT COUNT(*) FROM gkg_news").fetchone()[0]

print("\n" + "="*40)
print(f"🎉 恭喜！数据全部清洗并入库完成！")
print(f"🗄️ 数据库文件: {os.path.abspath(db_path)}")
print(f"📈 数据库内现存新闻总数: {total_records:,} 条")
print("="*40)

con.close()