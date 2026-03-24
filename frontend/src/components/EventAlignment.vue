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
            <p class="chart-hint">🔴 红色图钉 = 最新新闻事件 | 🟠 橙色图钉 = 价格波动触发点 | 💡 点击图钉查看详情</p>
          </div>
          <div ref="chartRef" class="chart-container"></div>
        </div>

        <!-- 新闻详情面板（图表下方） -->
        <div v-if="selectedMarkPoint" class="details-card">
          <div class="details-header">
            <h3 class="details-title">
              <span v-if="selectedMarkPoint.isSpike">⚡ 价格波动详情</span>
              <span v-else>📌 新闻事件详情</span>
            </h3>
            <button class="close-button" @click="selectedMarkPoint = null">✕</button>
          </div>
          
          <div v-if="selectedMarkPoint.isSpike" class="details-content spike-details">
            <!-- 波动信息 -->
            <div class="info-section">
              <div class="info-label">时间</div>
              <div class="info-value">{{ new Date(selectedMarkPoint.alignedX).toLocaleString() }}</div>
            </div>
            
            <div class="info-section">
              <div class="info-label">价格变化</div>
              <div class="info-value price-change">
                <span class="price-before">{{ (selectedMarkPoint.spike?.price_before || 0).toFixed(4) }}</span>
                <span class="arrow">→</span>
                <span class="price-after">{{ (selectedMarkPoint.spike?.price_after || 0).toFixed(4) }}</span>
                <span class="change-pct" :class="{ positive: selectedMarkPoint.spike?.price_change_pct > 0 }">
                  {{ selectedMarkPoint.spike?.price_change_pct > 0 ? '+' : '' }}{{ (selectedMarkPoint.spike?.price_change_pct * 100).toFixed(2) }}%
                </span>
              </div>
            </div>

            <!-- 相关新闻 -->
            <div v-if="selectedMarkPoint.relatedNews && selectedMarkPoint.relatedNews.length > 0" class="info-section news-list">
              <div class="info-label">💬 相关新闻（{{ selectedMarkPoint.relatedNews.length }}）</div>
              <div class="news-items">
                <div v-for="(news, idx) in selectedMarkPoint.relatedNews" :key="idx" class="news-item">
                  <div class="news-title-link">
                    <a v-if="news.url" :href="news.url" target="_blank" rel="noopener noreferrer" class="news-link">
                      {{ news.title }}
                    </a>
                    <span v-else>{{ news.title }}</span>
                  </div>
                  <div v-if="news.published_date" class="news-date">{{ formatDate(news.published_date) }}</div>
                  <div v-if="news.content" class="news-snippet">{{ news.content.substring(0, 150) }}...</div>
                </div>
              </div>
            </div>
            <div v-else class="empty-news">
              <p>未找到相关新闻</p>
            </div>
          </div>

          <div v-else class="details-content news-details">
            <!-- 新闻信息 -->
            <div class="info-section">
              <div class="info-label">发布时间</div>
              <div class="info-value">{{ new Date(selectedMarkPoint.alignedX).toLocaleString() }}</div>
            </div>

            <div class="info-section">
              <div class="info-label">标题</div>
              <div class="info-value news-title">
                <a v-if="selectedMarkPoint.newsInfo?.url" :href="selectedMarkPoint.newsInfo.url" target="_blank" rel="noopener noreferrer" class="news-link">
                  {{ selectedMarkPoint.newsInfo?.title }}
                </a>
                <span v-else>{{ selectedMarkPoint.newsInfo?.title }}</span>
              </div>
            </div>

            <div v-if="selectedMarkPoint.newsInfo?.content" class="info-section">
              <div class="info-label">摘要</div>
              <div class="info-value news-content">{{ selectedMarkPoint.newsInfo.content }}</div>
            </div>

            <div class="info-section">
              <div class="info-label">对齐概率</div>
              <div class="info-value probability">
                <div class="prob-bar">
                  <div class="prob-fill" :style="{ width: (selectedMarkPoint.alignedY * 100) + '%' }"></div>
                </div>
                <span class="prob-text">{{ (selectedMarkPoint.alignedY * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧：新闻时间轴（侧边栏） -->
      <div class="right-sidebar" v-if="newsList.length > 0 || spikeNewsList.length > 0">
        <div class="timeline-card">
          <!-- 标签页切换 -->
          <div class="tabs-header">
            <button 
              class="tab-button" 
              :class="{ active: currentTabInSidebar === 'news' }"
              @click="currentTabInSidebar = 'news'"
            >
              📰 最新新闻 ({{ newsList.length }})
            </button>
            <button 
              v-if="spikeNewsList.length > 0"
              class="tab-button" 
              :class="{ active: currentTabInSidebar === 'spikes' }"
              @click="currentTabInSidebar = 'spikes'"
            >
              ⚡ 波动新闻 ({{ spikeNewsList.length }})
            </button>
          </div>

          <!-- 新闻时间轴 -->
          <div v-if="currentTabInSidebar === 'news'" class="timeline">
            <div 
              v-for="(news, index) in newsList" 
              :key="'news-' + index" 
              class="timeline-item"
              :class="{ first: index === 0 }"
              @click="selectNewsFromTimeline(news)"
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

          <!-- 波动新闻时间轴 -->
          <div v-if="currentTabInSidebar === 'spikes'" class="timeline grouped-spikes">
            <div
              v-for="group in spikeNewsGroups"
              :key="group.groupKey"
              class="spike-group"
            >
              <button class="spike-group-header" @click="toggleSpikeGroup(group.groupKey)">
                <span class="spike-group-left">
                  <span class="spike-caret">{{ expandedSpikeGroups[group.groupKey] ? '▾' : '▸' }}</span>
                  <span>⚡ 波动时间: {{ group.timeLabel }}</span>
                </span>
                <span class="spike-group-right">
                  {{ group.items.length }} 条新闻
                </span>
              </button>

              <div v-if="expandedSpikeGroups[group.groupKey]" class="spike-group-body">
                <div
                  v-for="(news, index) in group.items"
                  :key="`${group.groupKey}-${index}`"
                  class="timeline-item spike"
                  :class="{ first: index === 0 }"
                  @click="selectSpikeNewsFromTimeline(news)"
                >
                  <div class="timeline-dot spike-dot"></div>
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
                    <div v-if="news.spikePrice" class="event-prob spike-price">
                      💰 概率: {{ (news.spikePrice * 100).toFixed(1) }}%
                    </div>
                  </div>
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
const spikeNewsList = ref([])
const spikeNewsGroups = ref([])
const expandedSpikeGroups = ref({})
const currentTabInSidebar = ref('news') // 'news' | 'spikes'
const selectedMarkPoint = ref(null) // 存储选中的标记点详情
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

  // 处理价格波动点
  const priceSpikeAlerts = Array.isArray(payload?.price_spike_alerts) ? payload.price_spike_alerts : []
  const spikeMarkPoints = priceSpikeAlerts
    .map((alert) => {
      const spike = alert?.spike
      if (!spike) return null
      
      const ts = Number(spike?.timestamp)
      const price = Number(spike?.price_after)
      if (!Number.isFinite(ts) || !Number.isFinite(price)) return null

      const relatedNews = Array.isArray(alert?.related_news) ? alert.related_news : []
      const newsContent = relatedNews
        .map((n) => n?.title || '')
        .filter(Boolean)
        .join('; ')

      return {
        coord: [ts * 1000, price],
        name: `价格波动: ${(spike?.price_before || 0).toFixed(4)} → ${(spike?.price_after || 0).toFixed(4)}`,
        newsInfo: {
          title: `价格波动 ${(spike?.price_change_pct * 100).toFixed(1)}%`,
          content: newsContent || '未找到相关新闻',
          url: ''
        },
        alignedX: ts * 1000,
        alignedY: price,
        isSpike: true,
        spike: spike,
        relatedNews: relatedNews
      }
    })
    .filter(Boolean)

  const allSpikeNews = priceSpikeAlerts.flatMap((alert) => {
    const spike = alert?.spike
    const relatedNews = Array.isArray(alert?.related_news) ? alert.related_news : []
    return relatedNews.map((newsItem) => ({
      ...normalizeNewsItem(newsItem),
      spikeTime: spike?.datetime || '',
      spikePrice: spike?.price_after || 0,
      spikeTimestamp: Number(spike?.timestamp) || 0,
      isPriceSpike: true
    }))
  })

  const groupedSpikes = priceSpikeAlerts
    .map((alert, groupIndex) => {
      const spike = alert?.spike
      if (!spike) return null

      const spikeTimestamp = Number(spike?.timestamp)
      const groupKey = `spike-${Number.isFinite(spikeTimestamp) ? spikeTimestamp : groupIndex}`
      const relatedNews = Array.isArray(alert?.related_news) ? alert.related_news : []

      const items = relatedNews
        .map((newsItem) => ({
          ...normalizeNewsItem(newsItem),
          spikeTime: spike?.datetime || '',
          spikePrice: spike?.price_after || 0,
          spikeTimestamp: Number(spike?.timestamp) || 0,
          isPriceSpike: true
        }))
        .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))

      const groupDate = spike?.datetime ? new Date(spike.datetime) : null
      const timeLabel = groupDate && !Number.isNaN(groupDate.getTime())
        ? groupDate.toLocaleString()
        : '时间未知'

      return {
        groupKey,
        spikeTimestamp: Number.isFinite(spikeTimestamp) ? spikeTimestamp : 0,
        timeLabel,
        items,
      }
    })
    .filter(Boolean)
    .sort((a, b) => b.spikeTimestamp - a.spikeTimestamp)

  return {
    marketSeries,
    news: enrichedNews,
    markPoints: [...markPoints, ...spikeMarkPoints],
    spikeAlerts: priceSpikeAlerts,
    spikeNews: allSpikeNews,
    spikeGroups: groupedSpikes,
    note: payload?.note || ''
  }
}

function toggleSpikeGroup(groupKey) {
  expandedSpikeGroups.value = {
    ...expandedSpikeGroups.value,
    [groupKey]: !expandedSpikeGroups.value[groupKey]
  }
}

function renderChart(timeSeries, markPoints) {
  if (!myChart) {
    myChart = echarts.init(chartRef.value)
  }

  // 添加唯一ID到每个标记点，用于点击事件识别
  const markPointsWithId = markPoints.map((point, index) => ({
    ...point,
    _pointId: index
  }))

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        if (params.componentType === 'markPoint') {
          const data = params.data
          const news = data.newsInfo
          const alignedTime = new Date(data.alignedX).toLocaleString()
          
          if (data.isSpike) {
            const spike = data.spike
            const priceChangePct = (spike?.price_change_pct * 100).toFixed(1)
            return `<div style="max-width: 350px; white-space: normal;">
                      <div style="font-weight:bold; color:#f59e0b; margin-bottom:5px;">⚡ 价格波动检测</div>
                      <div style="font-weight: bold; margin: 5px 0; color: #333;">${(spike?.price_before || 0).toFixed(4)} → ${(spike?.price_after || 0).toFixed(4)}</div>
                      <div style="font-size: 12px; color: #d97706; margin: 5px 0; font-weight: bold;">变化: +${priceChangePct}%</div>
                      <div style="border-top: 1px solid #fde68a; padding-top: 5px; margin-top: 8px; font-size: 11px; color: #92400e;">
                        <b>时间：</b> ${alignedTime}<br/>
                        <div style="margin-top: 4px; color: #b45309;">${news.content || '未找到相关新闻'}</div>
                      </div>
                    </div>`
          } else {
            const alignedProb = (data.alignedY * 100).toFixed(1)
            return `<div style="max-width: 350px; white-space: normal;">
                      <div style="font-weight:bold; color:#e74c3c; margin-bottom:5px;">📌 触发的新闻事件</div>
                      <div style="font-weight: bold; margin: 5px 0; color: #333;">${news.title}</div>
                      <div style="font-size: 12px; color: #666; margin: 5px 0;">${news.content}</div>
                      <div style="border-top: 1px solid #ddd; padding-top: 5px; margin-top: 8px; font-size: 11px; color: #999;">
                        <b>对齐坐标：</b> 时间 ${alignedTime}<br/>概率 <b style="color: #e74c3c;">${alignedProb}%</b>
                      </div>
                    </div>`
          }
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
            borderColor: '#fff',
            borderWidth: 2,
            shadowBlur: 8
          },
          label: { show: false },
          data: markPointsWithId.map((point) => {
            if (point.isSpike) {
              return {
                ...point,
                itemStyle: {
                  color: '#f59e0b',
                  borderColor: '#fff',
                  borderWidth: 2,
                  shadowBlur: 8,
                  shadowColor: 'rgba(245, 158, 11, 0.4)'
                }
              }
            } else {
              return {
                ...point,
                itemStyle: {
                  color: '#ef4444',
                  borderColor: '#fff',
                  borderWidth: 2,
                  shadowBlur: 8,
                  shadowColor: 'rgba(239, 68, 68, 0.4)'
                }
              }
            }
          })
        }
      }
    ]
  }

  myChart.setOption(option)

  // 添加点击事件处理
  myChart.off('click') // 移除之前的事件监听
  myChart.on('click', function (params) {
    const isMarkPoint =
      params?.componentType === 'markPoint' ||
      params?.componentSubType === 'markPoint' ||
      Number.isInteger(params?.data?._pointId)

    if (isMarkPoint && params.data) {
      const pointId = params.data._pointId
      const selectedPoint = markPointsWithId[pointId]
      if (selectedPoint) {
        selectedMarkPoint.value = selectedPoint
      }
    }
  })
}

function selectNewsFromTimeline(newsItem) {
  if (!newsItem) return
  if (newsItem.alignedPoint) {
    selectedMarkPoint.value = newsItem.alignedPoint
    return
  }

  selectedMarkPoint.value = {
    alignedX: newsItem.timestamp || Date.now(),
    alignedY: 0,
    newsInfo: {
      title: newsItem.title || '无标题新闻',
      content: newsItem.content || '暂无摘要',
      url: newsItem.url || ''
    },
    isSpike: false
  }
}

function selectSpikeNewsFromTimeline(newsItem) {
  if (!newsItem) return
  selectedMarkPoint.value = {
    alignedX: newsItem.spikeTime ? new Date(newsItem.spikeTime).getTime() : (newsItem.timestamp || Date.now()),
    alignedY: Number(newsItem.spikePrice) || 0,
    isSpike: true,
    spike: {
      price_before: Number(newsItem.spikePrice) || 0,
      price_after: Number(newsItem.spikePrice) || 0,
      price_change_pct: 0
    },
    relatedNews: [
      {
        title: newsItem.title || '无标题新闻',
        url: newsItem.url || '',
        content: newsItem.content || '暂无摘要',
        published_date: newsItem.date || ''
      }
    ],
    newsInfo: {
      title: newsItem.title || '无标题新闻',
      content: newsItem.content || '暂无摘要',
      url: newsItem.url || ''
    }
  }
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

    // 清空之前选中的标记点
    selectedMarkPoint.value = null

    if (result.note && result.marketSeries.length === 0) {
      alert(`提示: ${result.note}`)
    }

    newsList.value = result.news
    spikeNewsList.value = result.spikeNews
    spikeNewsGroups.value = result.spikeGroups || []
    expandedSpikeGroups.value = (result.spikeGroups || []).reduce((acc, group, index) => {
      acc[group.groupKey] = index === 0
      return acc
    }, {})
    currentTabInSidebar.value = result.spikeNews.length > 0 ? 'spikes' : 'news'

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

.tabs-header {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e5e7eb;
}

.tab-button {
  padding: 10px 16px;
  background: transparent;
  border: none;
  border-bottom: 3px solid transparent;
  font-size: 13px;
  font-weight: 600;
  color: #718096;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.tab-button.active {
  color: #667eea;
  border-bottom-color: #667eea;
}

.tab-button:hover:not(.active) {
  color: #4a5568;
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
  cursor: pointer;
  transition: all 0.3s ease;
}

.timeline-item.first .timeline-dot {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
}

.timeline-item.spike {
  background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
}

.timeline-item.spike.first .timeline-dot {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.2);
}

.timeline-item:hover {
  background: #edf2f7;
  transform: translateX(4px);
}

.timeline-item.spike:hover {
  background: linear-gradient(135deg, #fde68a 0%, #fcd34d 100%);
}

.grouped-spikes {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.spike-group {
  border: 1px solid #fcd34d;
  border-radius: 10px;
  background: #fffdf5;
  overflow: hidden;
}

.spike-group-header {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border: none;
  background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
  color: #92400e;
  font-size: 12px;
  font-weight: 700;
  padding: 10px 12px;
  cursor: pointer;
}

.spike-group-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.spike-caret {
  font-size: 12px;
  width: 12px;
  text-align: center;
}

.spike-group-right {
  font-size: 11px;
  color: #b45309;
}

.spike-group-body {
  padding: 10px;
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

.timeline-dot.spike-dot {
  background: #f59e0b;
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2);
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

.spike-trigger {
  font-size: 11px;
  color: #f59e0b;
  font-weight: 600;
  margin-bottom: 4px;
}

.spike-price {
  color: #f59e0b !important;
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

/* 新闻详情面板样式 */
.details-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  padding: 24px;
  margin-top: 20px;
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 12px;
}

.details-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  color: #1a202c;
}

.close-button {
  background: transparent;
  border: none;
  font-size: 20px;
  color: #a0aec0;
  cursor: pointer;
  padding: 4px 8px;
  transition: all 0.2s ease;
  border-radius: 4px;
}

.close-button:hover {
  background: #f0f4f8;
  color: #4a5568;
}

.details-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.details-content.spike-details {
  border-left: 4px solid #f59e0b;
  padding-left: 16px;
}

.details-content.news-details {
  border-left: 4px solid #ef4444;
  padding-left: 16px;
}

.info-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-label {
  font-size: 12px;
  font-weight: 700;
  color: #4a5568;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.info-value {
  font-size: 14px;
  color: #2d3748;
  line-height: 1.6;
}

.price-change {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.price-before {
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  font-weight: 600;
  color: #e53e3e;
  padding: 4px 8px;
  background: #fff5f5;
  border-radius: 4px;
}

.price-after {
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  font-weight: 600;
  color: #22863a;
  padding: 4px 8px;
  background: #f0fdf4;
  border-radius: 4px;
}

.arrow {
  color: #cbd5e0;
  font-weight: 600;
}

.change-pct {
  font-size: 13px;
  font-weight: 700;
  color: #e53e3e;
  padding: 4px 8px;
  background: #fff5f5;
  border-radius: 4px;
}

.change-pct.positive {
  color: #22863a;
  background: #f0fdf4;
}

.news-list {
  margin-top: 12px;
}

.news-items {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 8px;
}

.news-item {
  padding: 12px;
  background: #f7fafc;
  border-radius: 8px;
  border-left: 3px solid #667eea;
  transition: all 0.3s ease;
}

.news-item:hover {
  background: #edf2f7;
  transform: translateX(2px);
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
}

.news-title-link {
  font-size: 13px;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 4px;
}

.news-link {
  color: #667eea;
  text-decoration: none;
  transition: all 0.2s ease;
}

.news-link:hover {
  color: #4c51bf;
  text-decoration: underline;
}

.news-date {
  font-size: 11px;
  color: #718096;
  margin-bottom: 4px;
}

.news-snippet {
  font-size: 12px;
  color: #4a5568;
  line-height: 1.4;
}

.empty-news {
  text-align: center;
  padding: 20px;
  color: #a0aec0;
  font-size: 14px;
}

.news-title {
  font-size: 15px;
  font-weight: 600;
  color: #1a202c;
  line-height: 1.6;
}

.news-content {
  font-size: 13px;
  color: #4a5568;
  line-height: 1.6;
  max-height: 150px;
  overflow-y: auto;
  padding: 8px;
  background: #f7fafc;
  border-radius: 8px;
}

.probability {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.prob-bar {
  width: 100%;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
}

.prob-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px;
  transition: width 0.5s ease;
  box-shadow: 0 0 8px rgba(102, 126, 234, 0.4);
}

.prob-text {
  font-size: 14px;
  font-weight: 700;
  color: #667eea;
}
</style>
