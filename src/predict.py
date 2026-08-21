# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""统一推理入口（Omni-AD 校赛接口规范）。

用法（与规范第 7 节一致）::

    python -u src/predict.py \
        --data-root <测试数据根目录> \
        --manifest <eval_manifest.csv> \
        --model-dir <模型目录> \
        --output-dir <预测输出目录> \
        --device <cuda:0 或 cpu> \
        --num-workers 4

产物（写入 --output-dir）::

    <output-dir>/predictions.csv      # sample_id,image_score ∈ [0,1]
    <output-dir>/maps/<sample_id>.png # 单通道 16-bit PNG，与原图同尺寸
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from patchcore_lite import (
    build_backbone,
    load_category_ckpt,
    predict_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omni-AD 统一推理入口")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_model(model_dir: Path, device: torch.device):
    """加载共享主干 + 按类别懒加载 memory bank。"""
    model, features = build_backbone(device)

    shared = model_dir / "shared.pth"
    if not shared.exists():
        raise FileNotFoundError(
            f"模型目录缺少 shared.pth: {model_dir}（请检查 --model-dir）"
        )
    ckpt = torch.load(shared, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    banks: dict[str, dict] = {}

    def get_bank(category: str) -> dict:
        if category not in banks:
            ckpt_path = model_dir / "checkpoints" / f"{category}.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"缺少类别 {category} 的模型文件: {ckpt_path}"
                )
            banks[category] = load_category_ckpt(ckpt_path)
        return banks[category]

    return model, features, get_bank


def save_map_uint16(path: Path, anomaly_map: np.ndarray) -> None:
    """保存单通道 16-bit PNG（0~65535，与原图同尺寸）。"""
    from PIL import Image

    u16 = np.clip(np.round(np.asarray(anomaly_map, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)
    Image.fromarray(u16, mode="I;16").save(path)


def main() -> None:
    args = parse_args()
    set_seed(2026)  # 推理使用固定种子保证可复现
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    samples = read_manifest(args.manifest)
    model, features, get_bank = load_model(args.model_dir, device)

    from PIL import Image
    from torchvision import transforms

    to_tensor = transforms.ToTensor()

    predictions: list[tuple[str, float]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        category = sample["category"]
        image_path = args.data_root / sample["image_path"]

        img = Image.open(image_path).convert("RGB")
        image = to_tensor(img).to(device)

        score, anomaly_map = predict_image(
            model, features, image, get_bank(category), device
        )

        save_map_uint16(maps_dir / f"{sample_id}.png", anomaly_map)
        predictions.append((sample_id, score))
        print(f"[predict] {sample_id} ({category}) score={score:.6f}", flush=True)

    # predictions.csv
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image_score"])
        for sample_id, score in predictions:
            writer.writerow([sample_id, f"{score:.6f}"])

    print(f"[predict] done -> {args.output_dir} ({len(predictions)} samples)", flush=True)


if __name__ == "__main__":
    main()
