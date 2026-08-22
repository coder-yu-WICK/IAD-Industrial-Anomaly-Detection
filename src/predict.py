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

from patchcore import PatchCore


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
    """加载共享主干 + 按类别懒加载 memory bank。

    主干/层以 shared.pth 中记录的为准（load_shared 会自动重建匹配的架构），
    此处默认值仅作回退，与 train.py 的 ViT 配置保持一致。
    """
    model = PatchCore(
        device=device,
        backbone="vit_b_16",
        layers=("encoder.layers.2", "encoder.layers.3"),
    )

    shared = model_dir / "shared.pth"
    if not shared.exists():
        raise FileNotFoundError(
            f"模型目录缺少 shared.pth: {model_dir}（请检查 --model-dir）"
        )
    model.load_shared(shared)

    # 优先 ONNX 加速；缺失 .onnx 或 onnxruntime 不可用时回退 PyTorch 主干
    onnx_path = model_dir / "shared.onnx"
    if onnx_path.exists():
        try:
            model.load_onnx(onnx_path)
            print(f"[predict] 使用 ONNX 加速: {onnx_path}", flush=True)
        except Exception as exc:  # onnxruntime 未安装 / 版本不兼容等
            print(f"[predict] ONNX 加载失败({type(exc).__name__})，回退 PyTorch 推理", flush=True)
    else:
        print("[predict] 未找到 shared.onnx，使用 PyTorch 主干推理", flush=True)

    banks: dict[str, dict] = {}

    def get_bank(category: str) -> dict:
        if category not in banks:
            ckpt_path = model_dir / "checkpoints" / f"{category}.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"缺少类别 {category} 的模型文件: {ckpt_path}"
                )
            banks[category] = model.load_category(ckpt_path)
        return banks[category]

    return model, get_bank


def save_map_uint16(path: Path, anomaly_map: np.ndarray) -> None:
    """保存单通道 16-bit PNG（0~65535，与原图同尺寸）。

    ``Image.fromarray`` 对 uint16 自动推断为 I;16（不传 mode，避免 Pillow 13 移除参数）。
    """
    from PIL import Image

    u16 = np.clip(np.round(np.asarray(anomaly_map, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)
    Image.fromarray(u16).save(path)


def main() -> None:
    args = parse_args()
    set_seed(2026)  # 推理使用固定种子保证可复现
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    samples = read_manifest(args.manifest)
    model, get_bank = load_model(args.model_dir, device)

    from PIL import Image

    predictions: list[tuple[str, float]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        category = sample["category"]
        image_path = args.data_root / sample["image_path"]

        img = Image.open(image_path).convert("RGB")
        get_bank(category)
        pred = model.predict(img)

        save_map_uint16(maps_dir / f"{sample_id}.png", pred.anomaly_map)
        predictions.append((sample_id, pred.image_score))
        print(f"[predict] {sample_id} ({category}) score={pred.image_score:.6f}", flush=True)

    # predictions.csv
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image_score"])
        for sample_id, score in predictions:
            writer.writerow([sample_id, f"{score:.6f}"])

    print(f"[predict] done -> {args.output_dir} ({len(predictions)} samples)", flush=True)


if __name__ == "__main__":
    main()
