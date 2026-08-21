# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""统一训练入口（Omni-AD 校赛接口规范）。

用法（与规范第 6 节一致）::

    python -u src/train.py \
        --data-root <训练数据根目录> \
        --manifest <train_manifest.csv> \
        --output-dir <新模型输出目录> \
        --device <cuda:0 或 cpu> \
        --seed 2026 \
        --num-workers 4

产物（写入 --output-dir，不覆盖提交包内原始 model/）::

    <output-dir>/shared.pth                  # 共享主干 state_dict（离线加载必需）
    <output-dir>/checkpoints/<category>.pth   # 每类 memory bank
    <output-dir>/model_manifest.json          # 模型清单（hybrid 模式）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from patchcore_lite import (
    build_backbone,
    build_bank,
    save_category_ckpt,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omni-AD 统一训练入口")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读取训练清单（只含正常样本；不访问 test / ground_truth）
    samples = read_manifest(args.manifest)
    by_category: dict[str, list[Path]] = {}
    for row in samples:
        by_category.setdefault(row["category"], []).append(
            args.data_root / row["image_path"]
        )

    # 共享主干：离线加载必须把 state_dict 写入产物
    model, features = build_backbone(device)
    torch.save(
        {"state_dict": model.state_dict(), "seed": args.seed},
        args.output_dir / "shared.pth",
    )

    categories = sorted(by_category.keys())
    for category in categories:
        paths = by_category[category]
        print(f"[train] category={category} samples={len(paths)}", flush=True)
        bank_dict = build_bank(model, features, paths, device)
        save_category_ckpt(
            args.output_dir / "checkpoints" / f"{category}.pth", bank_dict, category
        )

    save_manifest(args.output_dir, categories, model_mode="hybrid")
    print(f"[train] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
