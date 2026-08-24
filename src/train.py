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


# 主干配置：默认 wide_resnet50_2（28×28 多尺度特征，像素级定位强于 ViT）。
# 换 ViT（Transformer 架构）跑对比：--backbone vit_b_16 --layers encoder.layers.2 encoder.layers.3
BACKBONE = "wide_resnet50_2"
LAYERS = ("layer2", "layer3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omni-AD 统一训练入口")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help="torchvision 主干名（默认 wide_resnet50_2，可选 vit_b_16）")
    parser.add_argument("--layers", type=str, nargs="+", default=list(LAYERS),
                        help="特征层名序列（默认 layer2 layer3）")
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
PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "model" / "pretrained"


def get_pretrained(backbone: str = BACKBONE) -> Path:
    """返回可用的预训练权重路径（按主干名）。

    首次在联网开发机上运行时会下载并缓存到 model/pretrained/，
    之后（含断网评测环境）直接读取本地文件。
    """
    file = PRETRAINED_DIR / f"{backbone}.pth"
    if file.exists():
        print(f"[train] 本地预训练权重: {file}", flush=True)
        return file

    print(f"[train] 未找到本地预训练权重，联网下载 {backbone} 并缓存 ...", flush=True)
    if backbone == "wide_resnet50_2":
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    elif backbone == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"不支持自动下载的主干 {backbone}，请手动放置 {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict()}, file)
    return file


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
        backbone=args.backbone,
        layers=tuple(args.layers),
        pretrained_path=get_pretrained(args.backbone),
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

    # ONNX 加速（可选）：训练末尾把截断主干导出为 .onnx，predict 端可用 onnxruntime 加速。
    # 缺 onnx/onnxscript 时静默跳过（不影响 shared.pth 与每类 bank 的正常产出）。
    try:
        onnx_path = args.output_dir / "shared.onnx"
        model.export_onnx(onnx_path, input_size=model.preprocess.crop_size)
        print(f"[train] onnx -> {onnx_path}", flush=True)
    except Exception as e:  # noqa: BLE001 —— 导出失败只降级，不中断训练
        print(f"[train] onnx 导出跳过（{type(e).__name__}: {e}）", flush=True)

    print(f"[train] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
