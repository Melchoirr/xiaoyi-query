"""
TimeRAGAgent — Layer 2: 意图解析 + 深度报告生成

职责:
    A. parse_intent()     — 调用 LLM 将自然语言查询解析为 Qdrant Filter 条件
    B. generate_report() — 调用 LLM 融合预测结果 + 召回元数据，生成可读性报告

LLM 选用: OpenAI gpt-4o-mini (速度快、成本低)，通过 OPENAI_API_KEY 环境变量认证。
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError


logger = logging.getLogger(__name__)


class IntentParseError(Exception):
    """意图解析失败"""

    pass


class ReportGenerationError(Exception):
    """报告生成失败"""

    pass


class TimeRAGAgent:
    """
    时序 RAG 智能 Agent

    核心能力:
        1. parse_intent(query)     -> Qdrant Filter dict
        2. generate_report(...)    -> 自然语言分析报告
    """

    # ------------------------------------------------------------------
    # Prompt 模板
    # ------------------------------------------------------------------

    INTENT_SYSTEM_PROMPT = """\
你是一个时序数据查询意图解析引擎。你的任务是将用户的自然语言描述**严格映射**为
Qdrant 向量数据库的 Filter 条件 JSON。

【可用 Payload 字段】
每个时序切片在 Qdrant 中存储了以下元数据:
  - month:        int,  1-12，事件发生月份
  - hour:         int,  0-23，事件发生小时
  - is_weekend:  bool, 是否周末（周六、周日）
  - time_of_day: str,  "night" | "morning" | "afternoon" | "evening"
    night     = 00:00-05:59
    morning   = 06:00-11:59
    afternoon = 12:00-17:59
    evening   = 18:00-23:59
  - source:       str,  数据源列名，如 "OT", "HUFL"
  - mu:          float, 历史序列的均值
  - sigma:       float, 历史序列的标准差

【Qdrant Filter JSON 结构（必须严格遵守）】
输出必须是一个 JSON 对象，至少包含 "must" 列表（可为空）：
{
  "must": [
    { "key": "<field>", "match": { "value": <bool|int|string> } },
    { "key": "<field>", "range": { "gte": <int>, "lt":  <int> } }
  ],
  "should": [],
  "must_not": []
}

【映射规则示例】
  "周末"            -> {"key": "is_weekend", "match": {"value": true}}
  "工作日"          -> {"key": "is_weekend", "match": {"value": false}}
  "夏季/七月"       -> {"key": "month",      "match": {"value": 7}}
  "用电高峰"        -> {"key": "time_of_day","match": {"value": "evening"}}
                     + {"key": "month",      "range": {"gte": 6, "lt": 9}}
  "深夜时段"        -> {"key": "time_of_day","match": {"value": "night"}}
  "OT 列"          -> {"key": "source",      "match": {"value": "OT"}}

【重要约束】
- 只输出 JSON，不要输出任何解释、markdown 或多余文字。
- 如果用户没有指定任何时间/来源过滤条件，返回 {"must": []}（不过滤）。
- 所有 key 必须在上述可用字段列表中，禁止凭空创造字段名。
- match.value 类型必须严格匹配字段类型（bool/int/string）。
"""

    INTENT_USER_TEMPLATE = '用户查询: "{query}"\n\n请输出 Filter JSON:'

    REPORT_SYSTEM_PROMPT = """\
你是一位时序数据领域的资深数据科学家，负责为用户生成专业的预测分析报告。

【输入信息】
- user_query:       用户的原始自然语言问题
- prediction:       XGBoost/IDW 融合后的预测数值序列（已反归一化，单位为真实物理量）
- retrieved_chunks:  从 Qdrant 检索到的 Top-K 历史相似片段，每条包含:
      - score:         相似度分数 (0~1, 越大越相似)
      - future_y:      该片段对应的未来真实值序列
      - time_features:  该片段的时间特征 {month, hour, is_weekend, time_of_day}

【报告结构要求】
请生成一份结构化分析报告，必须包含以下章节:

## 1. 查询意图解读
用 1-2 句话概括用户想了解的核心问题。

## 2. 预测结果
给出预测序列的统计摘要：
  - 预测值范围 [min, max]
  - 预测均值
  - 预测趋势（上升/下降/震荡）
  - 预测置信度说明（参考 Top-K 相似样本的一致性）

## 3. 历史相似样本分析
分析 Top-K 检索结果反映出的规律：
  - 相似样本的共同时间特征（如：主要集中在周末晚上）
  - 相似样本后续走势的共性（上升幅度、波动范围）
  - 与本次预测方向/幅度的一致或分歧程度

## 4. 专业建议
给出 2-3 条可操作的洞察或建议（如：监控阈值、异常预警、后续数据补充方向）。

【约束】
- 使用中文输出。
- 报告应专业、清晰、有洞见，不要泛泛而谈。
- 不要虚构任何具体数值，所有数字必须来自输入数据。
- prediction 和 retrieved_chunks 中的原始数据是你的分析依据，不要忽略任何一条。
"""

    REPORT_USER_TEMPLATE = """\
user_query: {user_query}

prediction: {prediction_json}

retrieved_chunks: {chunks_json}
"""

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        """
        Args:
            model:      LLM 模型名称，默认 gpt-4o-mini
            api_key:    OpenAI API Key，默认从环境变量 OPENAI_API_KEY 读取
            temperature: 生成温度，parse_intent 用 0.0（确定性），
                        generate_report 可在构造后通过参数覆盖
            timeout:    单次 API 调用超时（秒）
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请在环境变量中配置，或在构造 TimeRAGAgent 时传入 api_key 参数。"
            )

        self._client = OpenAI(api_key=resolved_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    # ------------------------------------------------------------------
    # 职责 A: 意图解析 — 自然语言 → Qdrant Filter
    # ------------------------------------------------------------------

    def parse_intent(self, query: str) -> Dict[str, Any]:
        """
        将用户的自然语言时序查询解析为 Qdrant Filter 条件。

        Args:
            query: 用户自然语言描述
                   示例: "结合夏季周末的用电高峰，预测未来趋势"
                         "周末晚上高油温会怎么走"

        Returns:
            Qdrant Filter dict，结构示例:
            {
                "must": [
                    {"key": "is_weekend",  "match": {"value": True}},
                    {"key": "time_of_day", "match": {"value": "evening"}},
                    {"key": "month",       "range": {"gte": 6, "lt": 9}},
                ],
                "should": [],
                "must_not": []
            }
            若无任何过滤条件，返回 {"must": []}。

        Raises:
            IntentParseError: LLM 未返回合法 Filter JSON
        """
        if not query or not query.strip():
            return {"must": [], "should": [], "must_not": []}

        messages = [
            {"role": "system", "content": self.INTENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.INTENT_USER_TEMPLATE.format(query=query.strip()),
            },
        ]

        raw_response = self._call_llm(
            messages=messages,
            temperature=0.0,
            stop=["```json", "```"],
        )

        return self._parse_filter_json(raw_response)

    # ------------------------------------------------------------------
    # 职责 B: 生成深度报告
    # ------------------------------------------------------------------

    def generate_report(
        self,
        user_query: str,
        fused_prediction: List[float],
        retrieved_metadata: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> str:
        """
        生成融合了数值预测和历史检索元数据的深度分析报告。

        Args:
            user_query:          用户的原始自然语言问题
            fused_prediction:    XGBoost/IDW 融合后的最终预测序列（已反归一化）
            retrieved_metadata:  Qdrant 召回的 Top-K 历史片段元数据列表，
                                每条应为 {"score": float, "payload": {...}} 结构
            temperature:         报告生成温度，默认 0.3（适度创造性）

        Returns:
            结构化中文分析报告字符串

        Raises:
            ReportGenerationError: LLM 调用失败或返回为空
        """
        # 整理检索结果为结构化 JSON
        chunks = []
        for item in retrieved_metadata:
            payload = item.get("payload", {})
            score = item.get("score", 0.0)
            chunks.append(
                {
                    "score": round(float(score), 4),
                    "future_y": _safe_list(payload.get("future_y", [])),
                    "time_features": {
                        "month": payload.get("month"),
                        "hour": payload.get("hour"),
                        "is_weekend": payload.get("is_weekend"),
                        "time_of_day": payload.get("time_of_day"),
                    },
                }
            )

        # 序列化预测序列（保留合理精度）
        pred_values = _safe_list(fused_prediction)

        prediction_summary = {
            "values": [round(v, 4) for v in pred_values],
            "length": len(pred_values),
            "min": round(float(min(pred_values)), 4) if pred_values else None,
            "max": round(float(max(pred_values)), 4) if pred_values else None,
            "mean": round(float(sum(pred_values) / len(pred_values)), 4)
            if pred_values
            else None,
        }

        user_content = self.REPORT_USER_TEMPLATE.format(
            user_query=user_query,
            prediction_json=json.dumps(
                {"prediction": prediction_summary}, ensure_ascii=False, indent=2
            ),
            chunks_json=json.dumps({"retrieved_chunks": chunks}, ensure_ascii=False, indent=2),
        )

        messages = [
            {"role": "system", "content": self.REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        report = self._call_llm(
            messages=messages,
            temperature=temperature,
            stop=None,
        )

        if not report or not report.strip():
            raise ReportGenerationError("LLM 返回了空内容，无法生成报告。")

        return report.strip()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        统一封装的 LLM 调用入口，包含重试和错误翻译。

        重试策略: RateLimitError 重试 3 次（指数退避），其余错误直接抛出。
        """
        import time

        max_retries = 3
        last_error: Exception = APIError(
            message="unknown", request=None, body=None
        )

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    stop=stop,
                    timeout=self._timeout,
                )
                content = response.choices[0].message.content
                if content is None:
                    content = ""
                return content

            except RateLimitError:
                last_error = RateLimitError("OpenAI API 触发速率限制")
                wait = 2 ** attempt
                logger.warning(
                    f"OpenAI RateLimitError，第 {attempt+1}/{max_retries} 次重试，等待 {wait}s"
                )
                time.sleep(wait)

            except APITimeoutError:
                last_error = APITimeoutError()
                logger.warning(
                    f"OpenAI API 超时（{self._timeout}s），第 {attempt+1}/{max_retries} 次重试"
                )
                time.sleep(1)

            except APIError as exc:
                last_error = exc
                logger.error(f"OpenAI API Error: {exc}")
                break

        # 所有重试均失败
        raise IntentParseError(
            f"LLM 调用失败（已重试 {max_retries} 次）: {last_error}"
        )

    @staticmethod
    def _parse_filter_json(raw: str) -> Dict[str, Any]:
        """
        从 LLM 返回的原始文本中提取并解析 JSON Filter。

        策略:
            1. 去掉 markdown 代码块包裹（```json ... ```）
            2. 尝试直接 json.loads
            3. 若失败，搜索第一个 '{' 到最后一个 '}' 的子串
            4. 若仍失败，抛出 IntentParseError
        """
        text = raw.strip()

        # 去掉 markdown 代码块
        for marker in ("```json", "```"):
            if text.startswith(marker):
                text = text[len(marker) :].strip()
            if text.endswith(marker):
                text = text[: -len(marker)].strip()

        # 尝试直接解析
        try:
            parsed = json.loads(text)
            # 校验基础结构
            if not isinstance(parsed, dict):
                raise IntentParseError(f"期望 JSON 对象，得到 {type(parsed).__name__}")
            if "must" not in parsed:
                parsed.setdefault("must", [])
            parsed.setdefault("should", [])
            parsed.setdefault("must_not", [])
            return parsed
        except json.JSONDecodeError:
            pass

        # 回退：定位 JSON 边界
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise IntentParseError(
                        f"JSON 边界提取后非对象: {type(parsed).__name__}"
                    )
                if "must" not in parsed:
                    parsed.setdefault("must", [])
                parsed.setdefault("should", [])
                parsed.setdefault("must_not", [])
                return parsed
            except json.JSONDecodeError as exc:
                raise IntentParseError(
                    f"无法从 LLM 响应中解析出合法 Filter JSON:\n"
                    f"--- raw ---\n{raw[:500]}\n--- error ---\n{exc}"
                )

        raise IntentParseError(
            f"LLM 响应中未找到有效的 JSON Filter 对象:\n{raw[:300]}"
        )


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _safe_list(value: Any) -> List[Any]:
    """安全地将任意对象转为列表（用于 JSON 序列化前的防御性处理）。"""
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None:
        return []
    return [value]
