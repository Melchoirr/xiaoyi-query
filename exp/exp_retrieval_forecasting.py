from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, Tuple

import numpy as np

from data_provider.data_factory import data_provider
from data_provider.window_builder import extract_hist_future_from_batch
from exp.exp_basic import Exp_Basic
from utils.logging_utils import get_logger
from utils.metrics import metric_dict, per_channel_metrics


class Exp_Retrieval_Forecasting(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.logger = get_logger()

    def _get_data(self, flag: str):
        return data_provider(self.args, flag)

    def _collect_windows(self, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        histories, futures, phases = [], [], []
        for batch_x, batch_y, batch_x_mark, _ in loader:
            hx, fy = extract_hist_future_from_batch(
                np.asarray(batch_x, dtype=np.float32),
                np.asarray(batch_y, dtype=np.float32),
                self.args.pred_len,
            )
            histories.append(hx)
            futures.append(fy)
            # phase uses last timestamp mark of query history
            phase = np.asarray(batch_x_mark, dtype=np.float32)[:, -1, :]
            phases.append(phase)
        return np.concatenate(histories, axis=0), np.concatenate(futures, axis=0), np.concatenate(phases, axis=0)

    def _distribution_stats(self, x: np.ndarray, name: str) -> Dict[str, float]:
        return {
            f"{name}_mean": float(np.mean(x)),
            f"{name}_std": float(np.std(x)),
            f"{name}_min": float(np.min(x)),
            f"{name}_max": float(np.max(x)),
        }

    def _diagnostics(self, pred_eval: np.ndarray, y_eval: np.ndarray) -> dict:
        diag = {}
        diag.update(self._distribution_stats(pred_eval, "pred"))
        diag.update(self._distribution_stats(y_eval, "true"))
        diag["per_channel_metrics"] = per_channel_metrics(pred_eval, y_eval, eps=self.args.mape_eps)

        dbg = getattr(self.model, "last_debug", {}) or {}
        if "candidate_scores" in dbg:
            scores = dbg["candidate_scores"]
            diag["topk_distance_mean"] = float(scores.mean())
            diag["topk_distance_std"] = float(scores.std())
            diag["topk_distance_min"] = float(scores.min())
            diag["topk_distance_max"] = float(scores.max())

        if "query_mean" in dbg and "candidate_hist_mean" in dbg:
            q_m = dbg["query_mean"][:, None, :]
            c_m = dbg["candidate_hist_mean"]
            diff = np.abs(c_m - q_m)
            diag["candidate_query_hist_mean_abs_diff"] = float(diff.mean())

        if dbg.get("query_phase") is not None and dbg.get("candidate_phase") is not None:
            q_p = dbg["query_phase"][:, None, :]
            c_p = dbg["candidate_phase"]
            pd = np.linalg.norm(c_p - q_p, axis=-1)
            diag["candidate_phase_l2_mean"] = float(pd.mean())

        if "agg_stats" in dbg:
            diag.update(dbg["agg_stats"])

        return diag

    def train(self, setting: str):
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")

        x_train, y_train, p_train = self._collect_windows(train_loader)
        x_val, y_val, p_val = self._collect_windows(val_loader)

        self.model.fit(x_train, y_train, train_phase=p_train)
        val_pred = self.model.forecast(x_val, query_phase=p_val)

        if self.args.eval_on_original_scale:
            val_pred_eval = val_data.inverse_transform(val_pred)
            y_val_eval = val_data.inverse_transform(y_val)
        else:
            val_pred_eval, y_val_eval = val_pred, y_val

        metrics = metric_dict(val_pred_eval, y_val_eval, mape_eps=self.args.mape_eps)
        self.logger.info("Validation metrics: %s", metrics)
        return metrics

    def test(self, setting: str) -> Dict[str, float]:
        test_data, test_loader = self._get_data("test")
        x_test, y_test, p_test = self._collect_windows(test_loader)

        start = time.time()
        pred = self.model.forecast(x_test, query_phase=p_test)
        runtime = time.time() - start

        if self.args.eval_on_original_scale:
            pred_eval = test_data.inverse_transform(pred)
            y_eval = test_data.inverse_transform(y_test)
        else:
            pred_eval, y_eval = pred, y_test

        metrics = metric_dict(pred_eval, y_eval, mape_eps=self.args.mape_eps)
        metrics["runtime"] = runtime
        diagnostics = self._diagnostics(pred_eval, y_eval)
        self._save_outputs(setting, pred_eval, y_eval, metrics, diagnostics)
        return metrics

    def _save_outputs(self, setting: str, pred: np.ndarray, true: np.ndarray, metrics: Dict[str, float], diagnostics: dict) -> None:
        out_dir = os.path.join(self.args.result_path, setting)
        report_dir = os.path.join(out_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "pred.npy"), pred)
        np.save(os.path.join(out_dir, "true.npy"), true)

        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])

        with open(os.path.join(report_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
