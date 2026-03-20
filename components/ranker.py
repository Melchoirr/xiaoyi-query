"""
排序与融合引擎 - IDW融合算法 + XGBoost微调
"""

from typing import List, Dict, Any
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge


class FusionRanker:
    """融合排序器"""

    def __init__(self):
        """初始化排序器"""
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42
        )
        self.ridge_model = Ridge(alpha=1.0)
        self._is_fitted = False

    def rank(
        self,
        search_results: List[Dict[str, Any]],
        future_length: int = 10
    ) -> np.ndarray:
        """
        对检索结果进行IDW融合排序

        Args:
            search_results: Qdrant返回的搜索结果列表
            future_length: 未来序列的长度

        Returns:
            融合后的预测序列
        """
        if not search_results:
            return np.zeros(future_length)

        weights = []
        future_ys = []
        features_list = []

        for result in search_results:
            score = result.get("score", 0.0)
            payload = result.get("payload", {})

            distance = 1.0 - score if score <= 1.0 else 0.0

            mu = payload.get("mu", 0.0)
            sigma = payload.get("sigma", 1.0)
            future_y = payload.get("future_y", [])

            if not future_y or len(future_y) < future_length:
                continue

            future_ys.append(future_y[:future_length])

            weight = 1.0 / (distance ** 2 + 1e-6)
            weights.append(weight)

            features_list.append([distance, mu, sigma])

        if not weights:
            return np.zeros(future_length)

        weights = np.array(weights)
        future_ys = np.array(future_ys)

        weights_normalized = weights / np.sum(weights)

        idw_prediction = np.sum(
            future_ys * weights_normalized[:, np.newaxis],
            axis=0
        )

        if self._is_fitted and len(features_list) > 1:
            features = np.array(features_list)
            targets = future_ys

            xgb_adjustment = self.xgb_model.predict(features)
            xgb_adjustment = xgb_adjustment.reshape(-1, 1)

            adjustment = xgb_adjustment.flatten() * 0.1

            idw_prediction = idw_prediction + adjustment

        return idw_prediction

    def fit(self, train_data: List[Dict[str, Any]]) -> None:
        """
        训练XGBoost微调模型

        Args:
            train_data: 训练数据列表，每项包含检索结果和真实值
        """
        if len(train_data) < 2:
            self._is_fitted = False
            return

        features_list = []
        targets_list = []

        for item in train_data:
            search_results = item.get("search_results", [])
            true_future = item.get("true_future", [])

            if not search_results or not true_future:
                continue

            for result in search_results:
                score = result.get("score", 0.0)
                payload = result.get("payload", {})

                distance = 1.0 - score if score <= 1.0 else 0.0
                mu = payload.get("mu", 0.0)
                sigma = payload.get("sigma", 1.0)

                features_list.append([distance, mu, sigma])
                targets_list.append(np.mean(np.array(true_future) - np.array(payload.get("future_y", true_future))))

        if len(features_list) > 1:
            X = np.array(features_list)
            y = np.array(targets_list)

            self.xgb_model.fit(X, y)
            self._is_fitted = True

    def inverse_distance_weighting(
        self,
        candidates: List[np.ndarray],
        distances: List[float],
        power: float = 2.0
    ) -> np.ndarray:
        """
        逆距离加权融合算法

        Args:
            candidates: 候选预测序列列表
            distances: 对应的距离/不相似度列表
            power: 距离的幂次，默认为2

        Returns:
            加权融合后的预测序列
        """
        if not candidates or not distances:
            return np.zeros(10)

        weights = []
        for dist in distances:
            if dist < 1e-6:
                weights.append(1e6)
            else:
                weights.append(1.0 / (dist ** power))

        weights = np.array(weights)
        weights_normalized = weights / np.sum(weights)

        candidates_matrix = np.array(candidates)
        fused_prediction = np.sum(
            candidates_matrix * weights_normalized[:, np.newaxis],
            axis=0
        )

        return fused_prediction
