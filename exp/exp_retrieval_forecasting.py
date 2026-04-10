from __future__ import annotations

import csv
import os
import time
from typing import Dict, Tuple

import numpy as np

from data_provider.data_factory import data_provider
from data_provider.window_builder import extract_hist_future_from_batch
from exp.exp_basic import Exp_Basic
from utils.logging_utils import get_logger
from utils.metrics import metric_dict


class Exp_Retrieval_Forecasting(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.logger = get_logger()

    def _get_data(self, flag: str):
        return data_provider(self.args, flag)

    def _collect_windows(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        histories, futures = [], []
        for batch_x, batch_y, _, _ in loader:
            hx, fy = extract_hist_future_from_batch(
                np.asarray(batch_x, dtype=np.float32),
                np.asarray(batch_y, dtype=np.float32),
                self.args.pred_len,
            )
            histories.append(hx)
            futures.append(fy)
        return np.concatenate(histories, axis=0), np.concatenate(futures, axis=0)

    def train(self, setting: str):
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")

        x_train, y_train = self._collect_windows(train_loader)
        x_val, y_val = self._collect_windows(val_loader)

        self.model.fit(x_train, y_train)
        val_pred = self.model.forecast(x_val)

        if self.args.eval_on_original_scale:
            val_pred_eval = val_data.inverse_transform(val_pred)
            y_val_eval = val_data.inverse_transform(y_val)
        else:
            val_pred_eval, y_val_eval = val_pred, y_val

        metrics = metric_dict(val_pred_eval, y_val_eval)
        self.logger.info("Validation metrics: %s", metrics)
        return metrics

    def test(self, setting: str) -> Dict[str, float]:
        test_data, test_loader = self._get_data("test")
        x_test, y_test = self._collect_windows(test_loader)

        start = time.time()
        pred = self.model.forecast(x_test)
        runtime = time.time() - start

        if self.args.eval_on_original_scale:
            pred_eval = test_data.inverse_transform(pred)
            y_eval = test_data.inverse_transform(y_test)
        else:
            pred_eval, y_eval = pred, y_test

        metrics = metric_dict(pred_eval, y_eval)
        metrics["runtime"] = runtime
        self._save_outputs(setting, pred_eval, y_eval, metrics)
        return metrics

    def _save_outputs(self, setting: str, pred: np.ndarray, true: np.ndarray, metrics: Dict[str, float]) -> None:
        out_dir = os.path.join(self.args.result_path, setting)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "pred.npy"), pred)
        np.save(os.path.join(out_dir, "true.npy"), true)

        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
