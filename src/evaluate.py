# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""本地评测脚本（开发用，不属于提交接口）。

按校赛规范（§8.3）官方指标计算，按类别计算再宏平均：
    Image-level AP / F1-max
    Pixel-level AP / F1-max / AUROC

用法::

    python src/evaluate.py \
        --predictions-dir work/predictions \
        --data-root data/Omni-AD-30-release \
        --manifest data/Omni-AD-30-release/test_manifest.csv

像素级指标用「分数直方图」流式累计（内存 O(bins)，与图片数量无关），
再反推 AP/AUC/F1-max，避免把全量像素同时载入内存（大图数据集直接 OOM）。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地指标评测")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--bins",
        type=int,
        default=4096,
        help="像素分数直方图分箱数（越大越接近全量精确，内存仍 O(1)；默认 4096）",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_scores(predictions_dir: Path) -> dict[str, float]:
    with (predictions_dir / "predictions.csv").open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: float(row["image_score"]) for row in csv.DictReader(f)}


def load_anomaly_map(predictions_dir: Path, sample_id: str) -> np.ndarray:
    """(H, W) float32，范围 [0,1]。"""
    arr = np.array(Image.open(predictions_dir / "maps" / f"{sample_id}.png")).astype(np.float32)
    return np.clip(arr / 65535.0, 0.0, 1.0)


def load_gt_mask(
    data_root: Path, image_path: str, map_shape: tuple[int, int], gt_rel: str | None = None
) -> tuple[np.ndarray, int]:
    """返回 (mask, label)。good 样本无掩膜，自动生成全零。

    优先用 manifest 的 ``ground_truth`` 列（精确相对路径）；缺省时回退按
    ``<category>/ground_truth/<defect>/<file>`` 目录结构推断。
    """
    parts = Path(image_path).parts
    category = parts[0]

    if parts[2] == "good":
        return np.zeros(map_shape, dtype=np.uint8), 0

    if gt_rel:
        gt_path = data_root / gt_rel
    else:
        gt_path = data_root / category / "ground_truth" / parts[2] / parts[-1]
    mask = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)
    return mask, 1


def image_f1_max(scores: np.ndarray, labels: np.ndarray) -> float:
    """图像级 F1-max：所有有效阈值上的最高 F1（精确，单次降序排序）。"""
    order = np.argsort(-scores, kind="stable")
    sl = labels[order]
    tp = np.cumsum(sl)
    fp = np.arange(1, len(sl) + 1) - tp
    fn = tp[-1] - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=np.float64), where=denom > 0)
    return float(f1.max())


def ap_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """图像级 AP；单类别（无正样本）时无定义，返回 nan 不参与宏平均。"""
    return average_precision_score(labels, scores) if len(set(labels)) > 1 else float("nan")


def pixel_metrics(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """由分数直方图反推像素级 (AUC, AP, F1-max)。pos/neg 为各 bin 计数，bin 0 = 分数最低。"""
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos == 0:
        return float("nan"), float("nan"), 0.0

    # 分数降序累计：tp/fp = 分数 ≥ 该 bin 的正/负像素数
    pos_d, neg_d = pos[::-1], neg[::-1]
    tp = np.cumsum(pos_d).astype(np.float64)
    fp = np.cumsum(neg_d).astype(np.float64)

    rec = tp / total_pos
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)

    # AP = Σ prec_k × (rec_k − rec_{k-1})（与 sklearn 口径一致）
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap = float(np.sum(prec * (rec - rec_prev)))

    # F1-max = max over 有效阈值 2PR/(P+R)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    f1max = float(f1.max())

    # AUC（梯形法，ROC 从 (0,0) 到 (1,1)）
    fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp)
    fpr_prev = np.concatenate([[0.0], fpr[:-1]])
    auc = float(np.sum((fpr - fpr_prev) * (rec + rec_prev) / 2.0))
    return auc, ap, f1max


def main() -> None:
    args = parse_args()
    samples = read_manifest(args.manifest)
    image_scores = load_scores(args.predictions_dir)
    nbins = args.bins

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)

    print(f"{'类别':<24}{'I-AP':>8}{'I-F1':>8}{'P-AP':>8}{'P-F1':>8}{'P-AUC':>8}")
    agg = {"I-AP": [], "I-F1": [], "P-AP": [], "P-F1": [], "P-AUC": []}
    for cat in sorted(by_cat):
        hist_pos = np.zeros(nbins, dtype=np.float64)
        hist_neg = np.zeros(nbins, dtype=np.float64)
        im_s: list[float] = []
        im_l: list[int] = []

        for s in by_cat[cat]:
            sid = s["sample_id"]
            score_map = load_anomaly_map(args.predictions_dir, sid)
            mask, label = load_gt_mask(
                args.data_root, s["image_path"], score_map.shape, s.get("ground_truth")
            )
            if mask.shape != score_map.shape:
                raise ValueError(f"{sid}: 掩膜尺寸 {mask.shape} 与热图 {score_map.shape} 不一致")

            pix_s = np.clip(score_map.ravel(), 0.0, 1.0)
            pix_l = mask.ravel()
            bins = (pix_s * (nbins - 1)).astype(np.int64)
            hist_pos += np.bincount(bins, weights=pix_l.astype(np.float64), minlength=nbins)
            hist_neg += np.bincount(bins, weights=(1.0 - pix_l).astype(np.float64), minlength=nbins)

            im_s.append(image_scores[sid])
            im_l.append(label)

        im_s = np.asarray(im_s, dtype=np.float32)
        im_l = np.asarray(im_l, dtype=np.int64)
        i_ap = ap_score(im_l, im_s)
        i_f1 = image_f1_max(im_s, im_l)
        p_auc, p_ap, p_f1 = pixel_metrics(hist_pos, hist_neg)

        agg["I-AP"].append(i_ap); agg["I-F1"].append(i_f1)
        agg["P-AP"].append(p_ap); agg["P-F1"].append(p_f1)
        agg["P-AUC"].append(p_auc)

        print(f"{cat:<24}{i_ap:8.4f}{i_f1:8.4f}{p_ap:8.4f}{p_f1:8.4f}{p_auc:8.4f}", flush=True)

    print("-" * 72)
    print(f"{'宏平均':<22}{np.nanmean(agg['I-AP']):8.4f}{np.nanmean(agg['I-F1']):8.4f}"
          f"{np.nanmean(agg['P-AP']):8.4f}{np.nanmean(agg['P-F1']):8.4f}"
          f"{np.nanmean(agg['P-AUC']):8.4f}")


if __name__ == "__main__":
    main()
