# 📈 Polymarket Event-Curve Aligner (预测市场事件与曲线对齐系统)

## 📖 项目背景

本项目为时序预测团队“预测结果整合”大任务的前置核心模块。
在预测市场（如 Polymarket）中，价格曲线的剧烈波动往往由现实世界的突发新闻（Events）驱动。传统时序模型难以捕捉这种基于离散文本信息的“Alpha”。

**本项目旨在：**
1. **获取与搜索**：通过大模型原生搜索引擎（Tavily AI）精准抓取特定时间窗口的结构化新闻。
2. **事件与曲线对齐（Alignment）**：在可视化层将 Polymarket 的时间序列曲线与现实新闻事件的时间戳进行锚定。
3. **为 LLM 调配提供上下文**：此对齐数据将作为后续输入，辅助 **LLM Orchestrator** 动态评判并分配基础时序模型（MOIRAI-agent）与传统机器学习模型（XGBoost）的融合权重。

## 🛠 技术栈 (Tech Stack)

### 前端 (Frontend)
- **框架**: Vue 3 (Composition API) + Vite
- **可视化**: Apache ECharts (用于绘制时序波动与标记对齐事件)
- **样式**: Tailwind CSS
- **网络请求**: Axios

### 后端 (Backend)
- **框架**: Python + FastAPI (提供高性能 RESTful API)
- **接口校验**: Pydantic
- **外部数据源**: Tavily AI API
- **架构模式**: Controller-Service-Client 经典分层架构

### 预测模型 (Roadmap / 后续整合)
- **MOIRAI-agent**: 零样本通用基础时序预测模型
- **XGBoost**: 基于统计与滞后特征的强监督基线模型
- **LLM**: 预测结果融合与动态权重调配大脑

---

## ✨ 核心功能 (Features)

- [x] **智能新闻检索**: 基于事件关键词与时间区间，调用 Tavily API 获取高相关度新闻。
- [x] **时序曲线模拟与渲染**: 动态生成/拉取 Polymarket 资产价格波动曲线（0~100% 概率）。
- [x] **高亮对齐 (Alignment)**: 将新闻事件以“图钉/气泡”的形式精准映射在价格波动的拐点上。
- [x] **交互式时间轴**: 鼠标悬浮图表拐点可预览新闻，图表下方配有完整的时间轴列表。

---

## 🚀 快速启动 (Quick Start)

### 1. 克隆项目

### 2. 后端服务运行 (FastAPI)
请确保你的机器已安装 Python 3.9+。

    cd backend
    pip install fastapi uvicorn pydantic tavily-python python-dotenv

请在 backend 目录下新建 .env 文件，并填入以下内容：
TAVILY_API_KEY=your_tavily_api_key_here

    uvicorn app.main:app --reload

也可以使用脚本文件直接运行：

    python run.py

(启动后可访问 http://localhost:8000/docs 查看自动生成的 API 文档)

### 3. 前端界面运行 (Vue 3 + Vite)
请确保你的机器已安装 Node.js (v16+)。

    cd frontend
    npm install
    npm install echarts
    npm run dev

(启动后在浏览器打开终端提示的本地地址，通常为 http://localhost:5173)

---

## 📂 项目目录结构 (Project Structure)

    polymarket-aligner/
    ├── backend/                  # Python 后端
    │   ├── app/
    │   │   ├── main.py           # FastAPI 入口与 CORS 配置
    │   │   ├── routers/          # API 路由定义 (Controller)
    │   │   ├── services/         # 业务逻辑与数据清洗 (Service)
    │   │   ├── clients/          # 第三方 API 调用封装 (Tavily)
    │   │   └── schemas/          # Pydantic 数据模型
    │   └── .env                  # 环境变量配置 (Git Ignore)
    │
    ├── frontend/                 # Vue 前端
    │   ├── src/
    │   │   ├── components/
    │   │   │   └── EventAlignment.vue  # 核心对齐可视化组件
    │   │   ├── App.vue           # 根组件
    │   │   └── main.js           # Vue 挂载入口
    │   ├── package.json
    │   └── vite.config.js
    │
    └── README.md                 # 项目说明文档

---

## 📅 下一步计划 (Next Steps & TODOs)

- [ ] **接入真实 Polymarket 数据**: 将当前 ECharts 的 Mock 曲线替换为 Polymarket Gamma API (CLOB) 提取的真实历史价格数据。
- [ ] **多模态特征工程**: 将对齐后的文本新闻转化为情绪分数（Sentiment Score），作为 XGBoost 的输入特征。
- [ ] **模型集成 (Ensemble)**: 部署 `uni-prophet/moirai` 模型提供时序 Baseline，并开发 LLM Prompt 模块进行双模型的预测融合与校准。
