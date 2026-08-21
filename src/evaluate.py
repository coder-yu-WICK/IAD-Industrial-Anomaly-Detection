# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""本地评测脚本（开发用，不属于提交接口）。

按校赛规范（§8.3）官方指标计算，按类别计算再宏平均：
    Image-level AP / F1-max
    Pixel-level AP / F1-max / AUROC

用法::

    python src/evaluate.py \
        --predictions-dir work/predictions \
        --data-root samples/Omni-AD-sample \
        --manifest samples/Omni-AD-sample/eval_manifest.csv

其中 --data-root 下需包含 ground_truth 目录（good 类样本自动视为全零掩膜）。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地指标评测")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_scores(predictions_dir: Path) -> dict[str, float]:
    with (predictions_dir / "predictions.csv").open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: float(row["image_score"]) for row in csv.DictReader(f)}


def load_anomaly_map(predictions_dir: Path, sample_id: str) -> np.ndarray:
    arr = np.array(Image.open(predictions_dir / "maps" / f"{sample_id}.png")).astype(np.float32)
    return arr / 65535.0  # [0,1]


def load_gt_mask(data_root: Path, image_path: str) -> tuple[np.ndarray, int]:
    """返回 (mask, label)。image_path 形如 <category>/test/<defect>/<file>。"""
    parts = Path(image_path).parts
    category, defect = parts[0], parts[2]
    img = Image.open(data_root / image_path)
    if defect == "good":
        label = 0
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    else:
        label = 1
        gt_path = data_root / category / "ground_truth" / defect / parts[-1]
        mask = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)
    return mask, label


def f1_max(scores: np.ndarray, labels: np.ndarray) -> float:
    """在所有有效阈值上取最高 F1（F1-max）。"""
    ts = np.linspace(scores.min(), scores.max(), 200)
    best = 0.0
    for t in ts:
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        best = max(best, f1)
    return best


def ap_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP；单类别（无正样本）时无定义，返回 nan 不参与宏平均。"""
    return average_precision_score(labels, scores) if len(set(labels)) > 1 else float("nan")


def main() -> None:
    args = parse_args()
    samples = read_manifest(args.manifest)
    image_scores = load_scores(args.predictions_dir)

    per_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        sid = s["sample_id"]
        mask, label = load_gt_mask(args.data_root, s["image_path"])
        score_map = load_anomaly_map(args.predictions_dir, sid)
        if score_map.shape != mask.shape:
            raise ValueError(f"{sid}: 热图尺寸 {score_map.shape} 与掩膜 {mask.shape} 不一致")

        cat = s["category"]
        per_cat[cat]["image_score"].append(image_scores[sid])
        per_cat[cat]["image_label"].append(label)
        per_cat[cat]["pixel_score"].append(score_map.ravel())
        per_cat[cat]["pixel_label"].append(mask.ravel())

    print(f"{'类别':<24}{'I-AP':>8}{'I-F1':>8}{'P-AP':>8}{'P-F1':>8}{'P-AUC':>8}")
    agg = {"I-AP": [], "I-F1": [], "P-AP": [], "P-F1": [], "P-AUC": []}
    for cat in sorted(per_cat):
        d = per_cat[cat]
        im_s = np.array(d["image_score"]); im_l = np.array(d["image_label"])
        px_s = np.concatenate(d["pixel_score"]); px_l = np.concatenate(d["pixel_label"])

        i_ap = ap_score(im_l, im_s)
        i_f1 = f1_max(im_s, im_l)
        p_ap = average_precision_score(px_l, px_s)
        p_f1 = f1_max(px_s, px_l)
        p_auc = roc_auc_score(px_l, px_s)

        agg["I-AP"].append(i_ap); agg["I-F1"].append(i_f1)
        agg["P-AP"].append(p_ap); agg["P-F1"].append(p_f1)
        agg["P-AUC"].append(p_auc)

        print(f"{cat:<24}{i_ap:8.4f}{i_f1:8.4f}{p_ap:8.4f}{p_f1:8.4f}{p_auc:8.4f}")

    print("-" * 72)
    print(f"{'宏平均':<22}{np.nanmean(agg['I-AP']):8.4f}{np.nanmean(agg['I-F1']):8.4f}"
          f"{np.nanmean(agg['P-AP']):8.4f}{np.nanmean(agg['P-F1']):8.4f}"
          f"{np.nanmean(agg['P-AUC']):8.4f}")


if __name__ == "__main__":
    main()
