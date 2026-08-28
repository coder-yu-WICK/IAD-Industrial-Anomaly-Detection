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

from patchcore import PatchCore, write_manifest


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


# 预训练权重随包存放（评测环境断网，禁止联网下载）
PRETRAINED_FILE = Path(__file__).resolve().parent.parent / "model" / "pretrained" / "franca_vitb14.pth"

# 主干配置：Franca（DINOv2 风格 ViT-B/14，patch 14，518 输入 → 37×37 网格），
# 取浅/中/深三个 Transformer block 特征。自写实现见 src/patchcore/vit.py。
BACKBONE = "franca_vitb14"
LAYERS = ("blocks.3", "blocks.6", "blocks.9")


def get_pretrained() -> Path:
    """返回随包的 Franca 预训练权重路径。

    权重由实现期一次性脚本 ``scripts/fetch_franca.py`` 在联网开发机拉取 Franca
    官方 hub 权重、对齐键名并前向校验后打包到 model/pretrained/。评测/复现环境
    断网，缺失即明确报错（禁止联网下载，更不能用 torchvision vit_b_16 顶替）。
    """
    if PRETRAINED_FILE.exists():
        print(f"[train] 本地预训练权重: {PRETRAINED_FILE}", flush=True)
        return PRETRAINED_FILE
    raise FileNotFoundError(
        f"缺少 Franca 预训练权重 {PRETRAINED_FILE}。\n"
        "请在联网开发机运行 `python scripts/fetch_franca.py` 生成后随包提交；\n"
        "注意：不能使用 torchvision vit_b_16.pth 顶替——franca_vitb14 是 patch=14 的 "
        "DINOv2 风格主干，键名/结构不同，会直接 size mismatch。"
    )


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
    model = PatchCore(
        device=device,
        backbone=BACKBONE,
        layers=LAYERS,
        pretrained_path=get_pretrained(),
    )
    model.save_shared(args.output_dir / "shared.pth", seed=args.seed)

    categories = sorted(by_category.keys())
    for category in categories:
        paths = by_category[category]
        print(f"[train] category={category} samples={len(paths)}", flush=True)
        bank_dict = model.fit(paths)
        model.save_category(
            args.output_dir / "checkpoints" / f"{category}.pth",
            category=category,
            bank_dict=bank_dict,
        )

    write_manifest(args.output_dir, categories, model_mode="hybrid")
    print(f"[train] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
