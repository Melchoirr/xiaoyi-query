<template>
  <div class="alignment-container">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">📊 事件对齐分析平台</h1>
        <p class="page-subtitle">实时监测预测市场价格与新闻事件的关联</p>
      </div>
    </div>

    <!-- 主容器 -->
    <div class="main-wrapper">
      <!-- 左侧：控制面板 + 图表 -->
      <div class="left-panel">
        <!-- 控制台 -->
        <div class="control-card">
          <h2 class="card-title">🔍 数据检索</h2>
          <div class="form-group">
            <label class="form-label">关键词检索</label>
            <input 
              v-model="query" 
              type="text" 
              placeholder="输入事件关键字..." 
              class="form-input"
              @keyup.enter="handleSearch"
            />
          </div>

          <div class="date-range-row">
            <div class="form-group date-group">
              <label class="form-label">开始日期</label>
              <input v-model="startDate" type="date" class="form-input" />
            </div>
            <div class="form-group date-group">
              <label class="form-label">结束日期</label>
              <input v-model="endDate" type="date" class="form-input" />
            </div>
          </div>
          
          <button 
            @click="handleSearch" 
            :disabled="loading"
            class="search-button"
          >
            <span v-if="loading" class="spinner"></span>
            <span>{{ loading ? '对齐分析中...' : '开始分析' }}</span>
          </button>
        </div>

        <!-- 图表区 -->
        <div class="chart-card">
          <div class="chart-header">
            <h2 class="card-title">📈 预测概率走势</h2>
            <p class="chart-hint">红色图钉 = 关键新闻事件</p>
          </div>
          <div ref="chartRef" class="chart-container"></div>
        </div>
      </div>

      <!-- 右侧：新闻时间轴（侧边栏） -->
      <div class="right-sidebar" v-if="newsList.length > 0">
        <div class="timeline-card">
          <h2 class="card-title timeline-title">📰 关键事件</h2>
          <div class="timeline">
            <div 
              v-for="(news, index) in newsList" 
              :key="index" 
              class="timeline-item"
              :class="{ first: index === 0 }"
            >
              <div class="timeline-dot"></div>
              <div class="timeline-content">
                <div class="event-time">{{ formatDate(news.date) }}</div>
                <a
                  v-if="news.url"
                  :href="news.url"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="event-title event-link"
                >
                  {{ news.title }}
                </a>
                <div v-else class="event-title">{{ news.title }}</div>
                <div v-if="news.alignedPoint" class="event-prob">
                  📊 {{ (news.alignedPoint.alignedY * 100).toFixed(1) }}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态提示 -->
    <div v-if="newsList.length === 0" class="empty-state">
      <div class="empty-icon">🎯</div>
      <p class="empty-text">点击"开始分析"按钮，查看事件对齐效果</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'

const query = ref('OpenAI 发布会')
const today = new Date()
const sevenDaysAgo = new Date(today)
sevenDaysAgo.setDate(today.getDate() - 7)

const formatDateInput = (d) => {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

const startDate = ref(formatDateInput(sevenDaysAgo))
const endDate = ref(formatDateInput(today))
const loading = ref(false)
const chartRef = ref(null)
const newsList = ref([])
let myChart = null
const EVENTS_API_URL = 'http://localhost:8000/api/events/alignment'

const formatDate = (dateStr) => {
  const date = new Date(dateStr)
  if (Number.isNaN(date.getTime())) return '时间未知'
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  return `${hours}:${minutes}`
}

function normalizeNewsItem(item) {
  const rawDate = item?.published_date || item?.date || ''
  const parsed = new Date(rawDate)
  const timestamp = Number.isNaN(parsed.getTime()) ? null : parsed.getTime()

  return {
    timestamp,
    date: rawDate || new Date().toISOString(),
    title: item?.title || '无标题新闻',
    content: item?.content || '暂无摘要',
    url: item?.url || ''
  }
}

function normalizeMarketSeries(series) {
  if (!Array.isArray(series)) return []
  return series
    .map((point) => {
      const ts = Number(point?.timestamp)
      const price = Number(point?.price)
      if (!Number.isFinite(ts) || !Number.isFinite(price)) return null
      return [ts * 1000, price]
    })
    .filter(Boolean)
    .sort((a, b) => a[0] - b[0])
}

async function fetchAlignmentFromBackend(keyword, start, end) {
  const params = new URLSearchParams({ query: keyword })
  if (start) params.set('start_date', start)
  if (end) params.set('end_date', end)
  params.set('news_limit', '15')
  params.set('fidelity', '300')

  const response = await fetch(`${EVENTS_API_URL}?${params.toString()}`, {
    method: 'GET'
  })

  if (!response.ok) {
    throw new Error(`获取对齐数据失败: ${response.status} ${response.statusText}`)
  }

  const payload = await response.json()
  const news = Array.isArray(payload?.news) ? payload.news : []
  const alignedEvents = Array.isArray(payload?.aligned_events) ? payload.aligned_events : []
  const marketSeries = normalizeMarketSeries(payload?.market_series)

  const normalizedNews = news
    .map(normalizeNewsItem)
    .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))

  const newsByKey = new Map()
  normalizedNews.forEach((item) => {
    const key = `${item.title}@@${item.url}`
    newsByKey.set(key, item)
  })

  const markPoints = alignedEvents
    .map((item) => {
      const marketTsSec = Number(item?.market_timestamp)
      const marketPrice = Number(item?.market_price)
      if (!Number.isFinite(marketTsSec) || !Number.isFinite(marketPrice)) return null

      const key = `${item?.title || ''}@@${item?.url || ''}`
      const baseNews = newsByKey.get(key)

      return {
        coord: [marketTsSec * 1000, marketPrice],
        name: item?.title || '事件',
        newsInfo: {
          title: item?.title || '无标题新闻',
          content: baseNews?.content || '',
          url: item?.url || ''
        },
        alignedX: marketTsSec * 1000,
        alignedY: marketPrice
      }
    })
    .filter(Boolean)

  const enrichedNews = normalizedNews.map((newsItem) => {
    const alignedPoint = markPoints.find((p) => p.newsInfo.title === newsItem.title && p.newsInfo.url === newsItem.url)
    return {
      ...newsItem,
      alignedPoint: alignedPoint || null
    }
  })

  return {
    marketSeries,
    news: enrichedNews,
    markPoints,
    note: payload?.note || ''
  }
}

function renderChart(timeSeries, markPoints) {
  if (!myChart) {
    myChart = echarts.init(chartRef.value)
  }

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        if (params.componentType === 'markPoint') {
          const news = params.data.newsInfo
          const alignedTime = new Date(params.data.alignedX).toLocaleString()
          const alignedProb = (params.data.alignedY * 100).toFixed(1)
          return `<div style="max-width: 350px; white-space: normal;">
                    <div style="font-weight:bold; color:#e74c3c; margin-bottom:5px;">📌 触发的新闻事件</div>
                    <div style="font-weight: bold; margin: 5px 0; color: #333;">${news.title}</div>
                    <div style="font-size: 12px; color: #666; margin: 5px 0;">${news.content}</div>
                    <div style="border-top: 1px solid #ddd; padding-top: 5px; margin-top: 8px; font-size: 11px; color: #999;">
                      <b>对齐坐标：</b> 时间 ${alignedTime}<br/>概率 <b style="color: #e74c3c;">${alignedProb}%</b>
                    </div>
                  </div>`
        } else {
          const date = new Date(params.data[0]).toLocaleString()
          const prob = (params.data[1] * 100).toFixed(1) + '%'
          return `${date}<br/>预测概率: <b>${prob}</b>`
        }
      }
    },
    grid: { left: '5%', right: '5%', bottom: '10%', top: '15%', containLabel: true },
    xAxis: {
      type: 'time',
      boundaryGap: false,
      splitLine: { show: true, lineStyle: { color: '#e5e7eb', type: 'dashed' } }
    },
    yAxis: {
      type: 'value',
      name: '发生概率',
      min: 0,
      max: 1,
      splitLine: { show: true, lineStyle: { color: '#f3f4f6' } },
      axisLabel: { formatter: (val) => (val * 100) + '%' }
    },
    series: [
      {
        name: 'Polymarket 概率',
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#3b82f6', width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59, 130, 246, 0.6)' },
            { offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
          ])
        },
        data: timeSeries,
        markPoint: {
          symbol: 'pin',
          symbolSize: [50, 60],
          itemStyle: { 
            color: '#ef4444',
            borderColor: '#fff',
            borderWidth: 2,
            shadowBlur: 8,
            shadowColor: 'rgba(239, 68, 68, 0.4)'
          },
          label: { show: false },
          data: markPoints
        }
      }
    ]
  }

  myChart.setOption(option)
}

const handleSearch = async () => {
  if (!query.value.trim()) return
  if (startDate.value && endDate.value && startDate.value > endDate.value) {
    alert('开始日期不能晚于结束日期')
    return
  }

  loading.value = true
  
  try {
    const result = await fetchAlignmentFromBackend(
      query.value.trim(),
      startDate.value,
      endDate.value
    )

    if (result.note && result.marketSeries.length === 0) {
      alert(`提示: ${result.note}`)
    }

    newsList.value = result.news

    if (!myChart) {
      initChart()
    }
    renderChart(result.marketSeries, result.markPoints)
  } catch (error) {
    console.error('Error in handleSearch:', error)
    alert('对齐数据获取失败，请确认后端服务已启动: http://localhost:8000')
  } finally {
    loading.value = false
  }
}

function initChart() {
  if (!chartRef.value) return
  
  const width = chartRef.value.clientWidth
  const height = chartRef.value.clientHeight
  
  if (width === 0 || height === 0) {
    setTimeout(() => initChart(), 300)
    return
  }
  
  if (myChart) myChart.dispose()
  
  myChart = echarts.init(chartRef.value)
  myChart.setOption({
    xAxis: { type: 'time' },
    yAxis: { type: 'value', min: 0, max: 1 },
    series: []
  })
}

onMounted(async () => {
  await nextTick()
  setTimeout(() => initChart(), 100)
  
  window.addEventListener('resize', () => {
    if (myChart) myChart.resize()
  })
})
</script>

<style scoped>
.alignment-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 20px;
}

.page-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px 20px;
  border-radius: 16px;
  margin-bottom: 30px;
  box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
}

.page-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 10px 0;
  letter-spacing: -0.5px;
}

.page-subtitle {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
  font-weight: 300;
}

.main-wrapper {
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 20px;
}

@media (max-width: 1200px) {
  .main-wrapper {
    grid-template-columns: 1fr;
  }
  
  .right-sidebar {
    grid-column: 1 / -1;
  }
}

.left-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.control-card,
.chart-card,
.timeline-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  padding: 24px;
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.control-card:hover,
.chart-card:hover,
.timeline-card:hover {
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 16px 0;
  color: #1a202c;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.date-range-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.date-group {
  min-width: 0;
}

.form-label {
  font-size: 14px;
  font-weight: 500;
  color: #4a5568;
}

.form-input {
  padding: 12px 16px;
  border: 2px solid #e2e8f0;
  border-radius: 10px;
  font-size: 14px;
  transition: all 0.3s ease;
  font-family: inherit;
}

.form-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-input::placeholder {
  color: #cbd5e0;
}

.search-button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.search-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

.search-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.chart-card {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chart-header {
  margin-bottom: 16px;
}

.chart-hint {
  font-size: 12px;
  color: #718096;
  margin: 6px 0 0 0;
}

.chart-container {
  width: 100%;
  height: 450px;
  min-height: 450px;
  border-radius: 8px;
}

.right-sidebar {
  position: sticky;
  top: 20px;
  height: fit-content;
  max-height: calc(100vh - 40px);
  overflow-y: auto;
}

.timeline-title {
  margin-bottom: 20px;
}

.timeline {
  position: relative;
}

.timeline-item {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  padding: 12px;
  border-radius: 8px;
  background: #f7fafc;
  transition: all 0.3s ease;
}

.timeline-item.first .timeline-dot {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
}

.timeline-item:hover {
  background: #edf2f7;
  transform: translateX(4px);
}

.timeline-dot {
  flex-shrink: 0;
  width: 12px;
  height: 12px;
  background: #e53e3e;
  border-radius: 50%;
  margin-top: 4px;
  box-shadow: 0 0 0 3px rgba(229, 62, 62, 0.2);
  transition: all 0.3s ease;
}

.timeline-content {
  flex: 1;
  min-width: 0;
}

.event-time {
  font-size: 12px;
  font-weight: 600;
  color: #667eea;
  margin-bottom: 4px;
}

.event-title {
  font-size: 13px;
  font-weight: 600;
  color: #1a202c;
  line-height: 1.4;
  word-break: break-word;
  margin-bottom: 4px;
}

.event-link {
  text-decoration: none;
}

.event-link:hover {
  color: #4c51bf;
  text-decoration: underline;
}

.event-prob {
  font-size: 11px;
  color: #e53e3e;
  font-weight: 500;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  background: white;
  border-radius: 16px;
  margin-top: 40px;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.empty-text {
  font-size: 16px;
  color: #718096;
  margin: 0;
}

.right-sidebar::-webkit-scrollbar {
  width: 6px;
}

.right-sidebar::-webkit-scrollbar-track {
  background: #f0f4f8;
  border-radius: 10px;
}

.right-sidebar::-webkit-scrollbar-thumb {
  background: #cbd5e0;
  border-radius: 10px;
  transition: background 0.3s;
}

.right-sidebar::-webkit-scrollbar-thumb:hover {
  background: #a0aec0;
}
</style>
