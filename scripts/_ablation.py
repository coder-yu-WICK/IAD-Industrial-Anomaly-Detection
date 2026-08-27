#!/usr/bin/env python
"""三个消融脚本共享的训练驱动（开发用，不进提交包）。

不调用 src/train.py，直接构造 PatchCore 并跑 fit/save，从而把每个消融的
"嵌套配置"完全收进脚本内部，保持 train.py（提交接口）干净不动。

默认参数对齐 README 联调示例：data/Omni-AD-30-release、seed=2026、cuda:0。
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from patchcore import PatchCore, write_manifest  # noqa: E402

PRETRAINED = ROOT / "model" / "pretrained" / "franca_vitb14.pth"
DATA_ROOT = ROOT / "data" / "Omni-AD-30-release"
MANIFEST = DATA_ROOT / "train_manifest.csv"
SEED = 2026
DEVICE = "cuda:0"
NUM_WORKERS = 8  # 仅示意；PatchCore 建 bank 本身为单进程逐图循环

# 高分辨率（518）Franca ViT-B/14 的通用配置
BACKBONE = "franca_vitb14"
LAYERS = ("blocks.3", "blocks.6", "blocks.9")
CASCADE_RATIOS = (0.1, 0.1)
MAX_EMBED = 50000


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


def run(
    out_rel: str,
    *,
    cascade: bool = True,
    use_prefix_dist: bool = False,
    prefix_dims: dict | None = None,
) -> None:
    """构造指定嵌套配置的 PatchCore，训练并落盘到 ROOT/out_rel。"""
    set_seed(SEED)
    device = torch.device(DEVICE)

    output_dir = ROOT / out_rel
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_manifest(MANIFEST)
    by_category: dict[str, list[Path]] = {}
    for row in samples:
        by_category.setdefault(row["category"], []).append(DATA_ROOT / row["image_path"])

    model = PatchCore(
        device=device,
        backbone=BACKBONE,
        layers=LAYERS,
        cascade_ratios=CASCADE_RATIOS,
        max_embed=MAX_EMBED,
        pretrained_path=PRETRAINED,
        use_prefix_dist=use_prefix_dist,
        prefix_dims=prefix_dims,
        cascade=cascade,
    )
    model.save_shared(output_dir / "shared.pth", seed=SEED)

    categories = sorted(by_category.keys())
    for category in categories:
        paths = by_category[category]
        print(f"[ablation] category={category} samples={len(paths)}", flush=True)
        bank = model.fit(paths)
        model.save_category(
            output_dir / "checkpoints" / f"{category}.pth",
            category=category,
            bank_dict=bank,
        )

    write_manifest(output_dir, categories, model_mode="hybrid")
    print(f"[ablation] done -> {output_dir}", flush=True)
