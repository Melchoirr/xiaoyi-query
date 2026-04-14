import duckdb
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 60)

DB_PATH = 'gdelt_master.duckdb'

def check_db_exists():
    if not os.path.exists(DB_PATH):
        print(f"❌ 找不到数据库文件: {DB_PATH}")
        exit()

def plot_trends(df, keyword):
    if df.empty:
        print("⚠️ 数据不足，无法绘图。")
        return

    # DuckDB 返回的已经是 datetime 对象，直接排序即可
    df = df.sort_values('day')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. 新闻量 (柱状图)
    ax1.bar(df['day'], df['article_count'], color='skyblue', alpha=0.7, label='新闻量')
    ax1.set_title(f"关键词 '{keyword}' 的全球新闻热度与情感趋势", fontsize=14)
    ax1.set_ylabel("报道篇数")
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. 平均情感 (折线图)
    ax2.plot(df['day'], df['avg_tone'], color='orange', marker='o', linewidth=2, label='平均情感分')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel("平均情感基调 (Tone)")
    ax2.set_xlabel("日期")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_name = f"trend_{keyword}_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(img_name)
    print(f"\n📊 统计图已生成并保存为: {img_name}")
    plt.show()

def run_pro_analysis(con):
    # 预检全库数据量
    total_count = con.execute("SELECT COUNT(*) FROM gkg_news").fetchone()[0]
    print(f"🔌 数据库连接成功！当前库内共有新闻: {total_count:,} 条")

    while True:
        print("\n" + "═"*60)
        print(" 🚀 GDELT 专业数据分析终端 (可视化版)")
        print("═"*60)
        print(" [1] 🔍 关键词快搜 (最新 10 条)")
        print(" [2] 📈 情感极性榜 (全库最值)")
        print(" [3] 🩺 数据库全面体检")
        print(" [4] 🎲 随机抽取样本")
        print(" [5] 📅 关键词时序趋势分析 (生成统计图)")
        print(" [0] 🚪 退出")
        print("═"*60)
        
        choice = input("请输入选择: ")
        
        if choice == '0': break

        elif choice == '5':
            kw = input("\n👉 输入分析关键词 (如 OpenAI): ")
            start_d = input("👉 开始日期 (YYYYMMDD，不限则直接回车): ")
            end_d = input("👉 结束日期 (YYYYMMDD，不限则直接回车): ")
            
            # 构建日期过滤条件 (针对 GKG 的字符串格式)
            date_filter = ""
            if start_d: date_filter += f" AND DATE >= '{start_d}000000'"
            if end_d:   date_filter += f" AND DATE <= '{end_d}235959'"

            # 🌟 修复点：使用 strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d') 转换日期
            sql_trend = f"""
                SELECT 
                    strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d') as day,
                    COUNT(*) as article_count,
                    AVG(CAST(split_part(V2Tone, ',', 1) AS FLOAT)) as avg_tone
                FROM gkg_news 
                WHERE ( Estimated_Title ILIKE '%{kw}%')
                {date_filter}
                GROUP BY day
                ORDER BY day
            """
            
            sql_preview = f"""
                SELECT DATE, Estimated_Title, SourceCommonName, CAST(split_part(V2Tone, ',', 1) AS FLOAT) as Tone
                FROM gkg_news
                WHERE ( Estimated_Title ILIKE '%{kw}%')
                {date_filter}
                ORDER BY Tone ASC
                LIMIT 5
            """
            
            try:
                start_t = time.perf_counter()
                df_trend = con.execute(sql_trend).df()
                df_preview = con.execute(sql_preview).df()
                elapsed = (time.perf_counter() - start_t) * 1000

                if df_trend.empty:
                    print(f"📭 未找到关于 '{kw}' 的时序数据，请检查关键词或日期范围。")
                    continue
                
                print(f"\n✅ 分析完成！匹配日期数: {len(df_trend)}，匹配文章总数: {df_trend['article_count'].sum():,}")
                print(f"⏱️  计算耗时: {elapsed:.2f} 毫秒")
                
                print("\n📌 该周期内最值得关注的低分新闻 (舆情风险)：")
                print(df_preview.to_string(index=False))
                
                # 绘图
                plot_trends(df_trend, kw)
            except Exception as e:
                print(f"❌ 查询出错: {e}")

        elif choice == '1':
            kw = input("\n👉 输入搜索关键词: ")
            sql = f"""
                SELECT DATE, Estimated_Title, SourceCommonName, CAST(split_part(V2Tone, ',', 1) AS FLOAT) AS Tone 
                FROM gkg_news 
                WHERE  Estimated_Title ILIKE '%{kw}%'
                ORDER BY DATE DESC LIMIT 10
            """
            start_t = time.perf_counter()
            df = con.execute(sql).df()
            elapsed = (time.perf_counter() - start_t) * 1000
            print(df.to_string(index=False))
            print(f"⏱️ 耗时: {elapsed:.2f} ms")

        # 其他选项简化处理...
        elif choice == '3':
            res = con.execute("SELECT COUNT(*), MIN(DATE), MAX(DATE) FROM gkg_news").fetchone()
            print(f"\n🩺 库内总数: {res[0]:,} 条 | 范围: {res[1]} 至 {res[2]}")

if __name__ == "__main__":
    check_db_exists()
    # read_only=True 模式更安全
    with duckdb.connect(DB_PATH, read_only=True) as con:
        run_pro_analysis(con)