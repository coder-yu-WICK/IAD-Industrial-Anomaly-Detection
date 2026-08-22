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
PRETRAINED_FILE = Path(__file__).resolve().parent.parent / "model" / "pretrained" / "vit_b_16.pth"

# 主干配置：torchvision 原生 ViT（vit_b_16），取第 2、3 个 Transformer block 特征
BACKBONE = "vit_b_16"
LAYERS = ("encoder.layers.2", "encoder.layers.3")


def get_pretrained() -> Path:
    """返回可用的预训练权重路径。

    首次在联网开发机上运行时会下载并缓存到 model/pretrained/，
    之后（含断网评测环境）直接读取本地文件。
    """
    if PRETRAINED_FILE.exists():
        print(f"[train] 本地预训练权重: {PRETRAINED_FILE}", flush=True)
        return PRETRAINED_FILE

    print("[train] 未找到本地预训练权重，联网下载并缓存到 model/pretrained/ ...", flush=True)
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    PRETRAINED_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict()}, PRETRAINED_FILE)
    return PRETRAINED_FILE


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
