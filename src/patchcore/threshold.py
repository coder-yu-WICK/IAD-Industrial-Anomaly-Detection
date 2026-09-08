# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""F1-adaptive 阈值：在带标签分数上扫描阈值取最大 F1（对齐 anomalib F1AdaptiveThreshold）。

注意：比赛指标（AP/AUROC/F1-max）由主办方在连续分数上计算，F1-max 本身即在评测时
重拟合阈值；本类仅用于可选的 pred_label/pred_mask 二值输出，不改变分数。
"""

from __future__ import annotations

import numpy as np


class F1AdaptiveThreshold:
    """拟合一个阈值 t，使 pred = score >= t 时分类 F1 最大。"""

    def __init__(self) -> None:
        self.threshold: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray, num_bins: int = 200) -> float:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.sum() == 0:
            self.threshold = float(scores.max())  # 无正样本：全部判正常
            return self.threshold
        if (labels == 0).sum() == 0:
            self.threshold = float(scores.min())  # 无负样本：全部判异常
            return self.threshold

        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, num_bins)))
        best_t, best_f1 = candidates[-1], -1.0
        for t in candidates:
            pred = scores >= t
            tp = float(((pred) & (labels == 1)).sum())
            fp = float(((pred) & (labels == 0)).sum())
            fn = float(((~pred) & (labels == 1)).sum())
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = float(best_t)
        return self.threshold

    @staticmethod
    def apply(scores: np.ndarray, threshold: float) -> np.ndarray:
        """二值化：>= threshold 判为 1，否则 0。"""
        return (np.asarray(scores) >= threshold).astype(np.uint8)
