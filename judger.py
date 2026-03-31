import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# ==========================================
# ⚙️ 配置区域
# ==========================================
INPUT_CSV = "polymarket_dataset_final.csv"

# 🌟 绘图配置
MAX_PLOTS = 5            # 最多绘制多少个市场的图表 (防止市场太多卡死)
SAVE_PLOTS = True        # 是否将图表保存为本地图片
SHOW_PLOTS = True        # 是否在运行中弹出窗口显示图表
PLOT_DIR = "market_plots" # 保存图表的文件夹名称

# 设置图表风格
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] =['Arial Unicode MS', 'SimHei'] # 兼容中文显示
plt.rcParams['axes.unicode_minus'] = False

def print_statistics(df):
    """
    在控制台打印详细的数据统计信息
    """
    print("\n" + "="*50)
    print("📊 Polymarket 数据集统计报告")
    print("="*50)
    
    # 1. 基础信息
    total_rows = len(df)
    start_date = df['datetime_utc'].min()
    end_date = df['datetime_utc'].max()
    print(f"🔹 [总计行数] : {total_rows:,} 行")
    print(f"🔹 [时间跨度] : {start_date.strftime('%Y-%m-%d %H:%M')} 至 {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # 2. 缺失值分析
    missing_prices = df['price'].isna().sum()
    missing_ratio = (missing_prices / total_rows) * 100
    print(f"🔹[价格缺失] : {missing_prices:,} 行 ({missing_ratio:.2f}%) - '如果是新市场且历史填充未覆盖则正常'")

    # 3. 维度统计
    num_keywords = df['keyword'].nunique()
    num_events = df['event_title'].nunique()
    num_markets = df['market_question'].nunique()
    num_tokens = df['token_id'].nunique()
    
    print("\n" + "-"*50)
    print("📈 结构化数据统计")
    print("-"*50)
    print(f"🔸 涉及关键词数 : {num_keywords}")
    print(f"🔸 独立事件总数 : {num_events}")
    print(f"🔸 独立市场总数 : {num_markets}")
    print(f"🔸 追踪Token数  : {num_tokens} (Outcomes)")

    # 4. 关键词分布
    print("\n" + "-"*50)
    print("🔍 按关键词 (Keyword) 的市场分布")
    print("-"*50)
    keyword_stats = df.groupby('keyword').agg(
        Events=('event_title', 'nunique'),
        Markets=('market_question', 'nunique'),
        Data_Points=('price', 'count')
    ).reset_index()
    print(keyword_stats.to_string(index=False))

    # 5. 市场活跃度排行 (数据点最多且非空的Top 5)
    print("\n" + "-"*50)
    print("🔥 数据最丰满的 Top 5 市场")
    print("-"*50)
    market_counts = df.dropna(subset=['price']).groupby('market_question').size().sort_values(ascending=False).head(5)
    for idx, (market, count) in enumerate(market_counts.items(), 1):
        print(f" {idx}. {market[:60]}... (有效数据点: {count})")
    
    print("="*50 + "\n")


def visualize_markets(df):
    """
    为市场生成时间序列可视化走势图
    """
    if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # 筛选出有效数据较多的市场来进行绘制
    # 这里我们选取有效价格数据最多的前 MAX_PLOTS 个市场
    valid_markets = df.dropna(subset=['price']).groupby('market_question').size().sort_values(ascending=False).head(MAX_PLOTS).index.tolist()
    
    if not valid_markets:
        print("❌ 没有足够的有效价格数据用于绘图。")
        return

    print(f"🎨 开始绘制 Top {len(valid_markets)} 个活跃市场的价格走势图...\n")

    for i, market in enumerate(valid_markets, 1):
        market_df = df[df['market_question'] == market].copy()
        
        # 使用 pivot_table 将不同 outcome 的价格转为不同的列
        pivot_df = pd.pivot_table(
            market_df, 
            index='datetime_utc', 
            columns='outcome', 
            values='price', 
            aggfunc='mean'
        )

        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制每条线 (Yes, No 等)
        for outcome in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[outcome], label=f"Outcome: {outcome}", linewidth=2)

        # 格式化图表
        event_title = market_df['event_title'].iloc[0]
        keyword = market_df['keyword'].iloc[0]
        
        plt.title(f"[{keyword.upper()}] {market}\nEvent: {event_title}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Datetime (UTC)", fontsize=12)
        plt.ylabel("Probability Price (0 to 1)", fontsize=12)
        
        # Y轴固定在 0-1 之间 (Polymarket概率区间)
        plt.ylim(-0.05, 1.05)
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5) # 添加0.5概率辅助线
        
        # X轴时间格式化
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate() # 自动旋转日期标签防止重叠

        plt.legend(title="Outcomes", loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # 保存或展示图表
        if SAVE_PLOTS:
            safe_filename = "".join([c for c in market[:50] if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            filename = f"{PLOT_DIR}/Plot_{i}_{safe_filename}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"✅ 图表已保存: {filename}")

        if SHOW_PLOTS:
            plt.show()
        
        plt.close() # 释放内存

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到数据文件: {INPUT_CSV}")
        print("请先运行爬虫脚本生成数据集。")
        sys.exit(1)

    print(f"⏳ 正在加载数据集 {INPUT_CSV} ...")
    try:
        # 读取数据并将时间列转换为 datetime 对象
        df = pd.read_csv(INPUT_CSV, parse_dates=['datetime_utc'])
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        sys.exit(1)

    # 1. 输出统计报告
    print_statistics(df)

    # 2. 生成图表
    visualize_markets(df)
    
    print("\n🎉 所有分析与可视化任务完成！")

if __name__ == "__main__":
    main()