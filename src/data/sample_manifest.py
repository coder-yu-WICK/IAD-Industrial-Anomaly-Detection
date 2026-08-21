# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""构造 Omni-AD-30 训练 / 测试 manifest。

在数据根目录下生成两个 CSV，供本地训练 / 推理 / 评测使用::

    python -u src/data/sample_manifest.py \
        --data-root data/Omni-AD-30-release

产物（写入 --data-root 目录）::

    train_manifest.csv    sample_id, category, image_path
    test_manifest.csv     sample_id, category, image_path, anomaly, ground_truth

字段说明:
    sample_id     由 <类别>_<缺陷目录>_<图片编号> 组成（train 无缺陷目录层），
                  全局唯一，可直接作为 predict.py 的 maps/<sample_id>.png 文件名。
    anomaly       仅 test：bool，False=正常样本（test/good），True=异常样本。
    ground_truth  仅 test：异常样本对应的掩膜相对路径，正常样本留空。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_FIELDS = ["sample_id", "category", "image_path"]
TEST_FIELDS = ["sample_id", "category", "image_path", "anomaly", "ground_truth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 Omni-AD-30 train/test manifest")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "Omni-AD-30-release",
        help="数据根目录（扫描其下的类别目录），默认 data/Omni-AD-30-release",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def img_key(path: Path) -> tuple[int, str]:
    """按 (文件名前缀数字, 文件名) 排序，保证 002 排在 010 之前。"""
    m = re.match(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 0, path.name)


def make_sample_id(category: str, folder: str | None, stem: str) -> str:
    """全局唯一 sample_id：train 为 <category>_<编号>，test 为 <category>_<folder>_<编号>。"""
    if folder is None:
        return f"{category}_{stem}"
    return f"{category}_{folder}_{stem}"


def scan_category(root: Path, category: str) -> tuple[list[dict], list[dict]]:
    """扫描单个类别，返回 (train_rows, test_rows)。缺失目录时告警并跳过。"""
    cat_dir = root / category
    train_good = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    # 1. train：good 文件夹内的全部图片，编号作为 sample_id
    train_rows: list[dict] = []
    if train_good.is_dir():
        for img in sorted(train_good.iterdir(), key=img_key):
            if not is_image(img):
                continue
            train_rows.append(
                {
                    "sample_id": make_sample_id(category, None, img.stem),
                    "category": category,
                    "image_path": img.relative_to(root).as_posix(),
                }
            )
    else:
        print(f"[warn] {category}: 缺少 train/good，跳过: {train_good}", flush=True)

    if not test_dir.is_dir():
        print(f"[warn] {category}: 缺少 test，跳过: {test_dir}", flush=True)
        return train_rows, []

    # 2. test/good：正常样本，anomaly=False，无 ground_truth
    test_rows: list[dict] = []
    good_dir = test_dir / "good"
    if good_dir.is_dir():
        for img in sorted(good_dir.iterdir(), key=img_key):
            if not is_image(img):
                continue
            test_rows.append(
                {
                    "sample_id": make_sample_id(category, "good", img.stem),
                    "category": category,
                    "image_path": img.relative_to(root).as_posix(),
                    "anomaly": False,
                    "ground_truth": "",
                }
            )

    # 3. test 其余缺陷文件夹（含 combined）：异常样本，anomaly=True，取对应 ground_truth
    defect_dirs = sorted(
        (d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"),
        key=lambda p: p.name,
    )
    for defect_dir in defect_dirs:
        defect = defect_dir.name
        gt_folder = gt_dir / defect
        for img in sorted(defect_dir.iterdir(), key=img_key):
            if not is_image(img):
                continue
            gt_rel = ""
            gt_file = gt_folder / img.name
            if gt_file.is_file():
                gt_rel = gt_file.relative_to(root).as_posix()
            else:
                print(
                    f"[warn] {category}: 缺 ground_truth，留空: {gt_file}",
                    flush=True,
                )
            test_rows.append(
                {
                    "sample_id": make_sample_id(category, defect, img.stem),
                    "category": category,
                    "image_path": img.relative_to(root).as_posix(),
                    "anomaly": True,
                    "ground_truth": gt_rel,
                }
            )

    return train_rows, test_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--data-root 不存在: {root}")

    # 仅扫描一级子目录作为类别（.xlsx 等文件与非目录自然被排除）
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"[manifest] root={root} categories={len(categories)}", flush=True)

    train_all: list[dict] = []
    test_all: list[dict] = []
    for category in categories:
        train_rows, test_rows = scan_category(root, category)
        train_all.extend(train_rows)
        test_all.extend(test_rows)
        n_good = sum(not r["anomaly"] for r in test_rows)
        n_anomaly = sum(r["anomaly"] for r in test_rows)
        print(
            f"[manifest] {category:<26} train={len(train_rows):>4} "
            f"test_good={n_good:>4} test_anomaly={n_anomaly:>4}",
            flush=True,
        )

    write_csv(root / "train_manifest.csv", train_all, TRAIN_FIELDS)
    write_csv(root / "test_manifest.csv", test_all, TEST_FIELDS)
    print(f"[manifest] done -> {root / 'train_manifest.csv'} ({len(train_all)} rows)", flush=True)
    print(f"[manifest] done -> {root / 'test_manifest.csv'} ({len(test_all)} rows)", flush=True)


if __name__ == "__main__":
    main()
