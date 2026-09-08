# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore + Segment Anything 并行推理入口。"""
from __future__ import annotations

import argparse
import csv
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from patchcore import PatchCore

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore + SAM 统一推理入口")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, required=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--category", action="append", default=None,
                   help="只推理指定类别；可重复传入，默认全部类别")
    p.add_argument("--sam-checkpoint", type=Path, default=None)
    p.add_argument("--sam-model-type", choices=tuple(SAM_CHECKPOINT_URLS), default="vit_b")
    p.add_argument("--sam-iou-threshold", type=float, default=0.35,
                   help="PatchCore 二值区域与 SAM 区域的 mIoU 阈值")
    p.add_argument("--patch-threshold", type=float, default=0.55,
                   help="PatchCore 热图二值化阈值")
    p.add_argument("--sam-surrounding-decay", type=float, default=0.35,
                   help="兼容旧参数：SAM 边界融合强度，0 不调整，1 完全贴合等值线")
    p.add_argument("--sam-boundary-width", type=int, default=5,
                   help="SAM 轮廓向内外平滑的像素宽度")
    p.add_argument("--disable-sam", action="store_true", help="仅用于无 SAM 依赖环境的回退测试")
    p.add_argument("--visualize-category", type=str, default=None,
                   help="输出该类别的原图/PatchCore/SAM/融合可视化，不指定则不输出")
    p.add_argument("--visualize-dir", type=Path, default=None,
                   help="可视化图片目录，默认 <output-dir>/visualizations")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def load_model(model_dir: Path, device: torch.device):
    model = PatchCore(device=device, backbone="dinov2_vitl14",
                      layers=("blocks.6", "blocks.12", "blocks.18"))
    shared = model_dir / "shared.pth"
    if not shared.exists(): raise FileNotFoundError(f"模型目录缺少 shared.pth: {model_dir}")
    model.load_shared(shared)
    onnx_path = model_dir / "shared.onnx"
    if onnx_path.exists():
        try: model.load_onnx(onnx_path); print(f"[predict] 使用 ONNX 加速: {onnx_path}", flush=True)
        except Exception as exc: print(f"[predict] ONNX 回退 PyTorch: {exc}", flush=True)
    banks: dict[str, dict] = {}
    def get_bank(category: str) -> dict:
        if category not in banks:
            path = model_dir / "checkpoints" / f"{category}.pth"
            if not path.exists(): raise FileNotFoundError(f"缺少类别模型文件: {path}")
            banks[category] = model.load_category(path)
        return banks[category]
    return model, get_bank


def ensure_sam_checkpoint(path: Path, model_type: str) -> Path:
    """下载 SAM 权重到本地；已有文件时完全离线运行。"""
    if path.exists(): return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = SAM_CHECKPOINT_URLS[model_type]
    print(f"[predict] 下载 SAM 权重: {url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_sam(args: argparse.Namespace, device: torch.device):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("SAM 模式需要安装 segment-anything（pip install git+https://github.com/facebookresearch/segment-anything.git）") from exc
    checkpoint = args.sam_checkpoint or (args.model_dir / "sam" / f"sam_{args.sam_model_type}.pth")
    ensure_sam_checkpoint(checkpoint, args.sam_model_type)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint)).to(device)
    sam.eval()
    return SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86,
                                     stability_score_thresh=0.90, min_mask_region_area=64)


def sam_mask_and_miou(generator, image: np.ndarray, patch_map: np.ndarray, threshold: float):
    """选取与 PatchCore 区域最相似的 SAM mask，并返回 mIoU。"""
    patch_binary = patch_map >= threshold
    masks = generator.generate(image)
    if not masks: return np.zeros(patch_map.shape, dtype=bool), 0.0
    best_mask, best_iou = np.zeros(patch_map.shape, bool), 0.0
    for item in masks:
        mask = np.asarray(item["segmentation"], dtype=bool)
        inter = np.logical_and(mask, patch_binary).sum()
        union = np.logical_or(mask, patch_binary).sum()
        iou = float(inter / union) if union else 0.0
        if iou > best_iou: best_mask, best_iou = mask, iou
    return best_mask, best_iou


def fuse_map(patch_map: np.ndarray, sam_mask: np.ndarray, miou: float,
             iou_threshold: float, surrounding_decay: float,
             boundary_width: int = 5) -> np.ndarray:
    """用 SAM 轮廓调整 PatchCore 等值线，不把整块 SAM 区域变成异常。

    SAM 只在轮廓窄带内作为空间先验：轮廓内外的 PatchCore 等值线向 SAM
    边界平滑过渡；SAM mask 内部仍保留原始 PatchCore 分数，因此不会让
    正常物体整块变成 1.0。
    """
    patch = np.clip(np.asarray(patch_map, dtype=np.float32), 0.0, 1.0)
    if miou < iou_threshold or not np.any(sam_mask):
        return patch.copy()

    # scipy 是可选依赖；无 scipy 时退化为原图，避免改变 PatchCore 基线。
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except ImportError:
        return patch.copy()

    mask = np.asarray(sam_mask, dtype=bool)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    width = max(1, int(boundary_width))
    # t=0 在 SAM 外侧，t=1 在 SAM 内侧，等值线以 sigmoid 平滑。
    signed = inside - outside
    weight = np.clip(0.5 + signed / (2.0 * width), 0.0, 1.0)
    weight = gaussian_filter(weight.astype(np.float32), sigma=max(0.5, width / 3.0))
    weight = np.clip(weight, 0.0, 1.0)

    # 仅把 PatchCore 的局部高分向轮廓轻微扩展/收缩，绝不注入固定高分。
    # surrounding_decay=0 时不调整；值越大，等值线约束越强。
    strength = float(np.clip(surrounding_decay, 0.0, 1.0))
    boundary_target = gaussian_filter(patch.astype(np.float32), sigma=max(0.5, width / 2.0))
    fused = patch * (1.0 - strength * weight) + boundary_target * (strength * weight)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def save_map_uint16(path: Path, anomaly_map: np.ndarray) -> None:
    u16 = np.clip(np.round(np.asarray(anomaly_map) * 65535), 0, 65535).astype(np.uint16)
    Image.fromarray(u16).save(path)


def save_visualization(path: Path, image: Image.Image, patch_map: np.ndarray,
                       sam_mask: np.ndarray, fused_map: np.ndarray, miou: float) -> None:
    """保存四联图：原图、PatchCore、SAM mask、融合热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    rgb = np.asarray(image)
    axes[0].imshow(rgb); axes[0].set_title("Original")
    axes[1].imshow(rgb); axes[1].imshow(patch_map, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("PatchCore heatmap")
    axes[2].imshow(rgb); axes[2].imshow(sam_mask, cmap="spring", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(f"SAM mask (mIoU={miou:.3f})")
    axes[3].imshow(rgb); axes[3].imshow(fused_map, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[3].set_title("Fused heatmap")
    for ax in axes: ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args(); set_seed(2026); device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "maps").mkdir(exist_ok=True)
    samples = read_manifest(args.manifest)
    if args.category: samples = [s for s in samples if s["category"] in set(args.category)]
    if args.visualize_category:
        samples = [s for s in samples if s["category"] == args.visualize_category]
        if not samples:
            raise ValueError(f"可视化类别不存在或没有样本: {args.visualize_category}")
    visualize_dir = args.visualize_dir or (args.output_dir / "visualizations")
    model, get_bank = load_model(args.model_dir, device)
    generator = None if args.disable_sam else load_sam(args, device)
    predictions = []
    for sample in samples:
        sid, category = sample["sample_id"], sample["category"]
        image = Image.open(args.data_root / sample["image_path"]).convert("RGB")
        get_bank(category)
        # 先并行提取两路结果，再用 PatchCore 热图计算 SAM 的 mIoU。
        # SAM 生成不依赖 PatchCore，因此两条推理路径确实同时启动。
        with ThreadPoolExecutor(max_workers=2) as pool:
            patch_future = pool.submit(model.predict, image)
            sam_future = (pool.submit(generator.generate, np.asarray(image)) if generator is not None else None)
            patch_pred = patch_future.result()
            if sam_future:
                sam_items = sam_future.result()
                patch_binary = patch_pred.anomaly_map >= args.patch_threshold
                best_mask, miou = np.zeros(patch_binary.shape, bool), 0.0
                for item in sam_items:
                    mask = np.asarray(item["segmentation"], dtype=bool)
                    inter = np.logical_and(mask, patch_binary).sum()
                    union = np.logical_or(mask, patch_binary).sum()
                    iou = float(inter / union) if union else 0.0
                    if iou > miou: best_mask, miou = mask, iou
                sam_mask = best_mask
            else:
                sam_mask, miou = np.zeros(patch_pred.anomaly_map.shape, bool), 0.0
        fused = fuse_map(
            patch_pred.anomaly_map, sam_mask, miou, args.sam_iou_threshold,
            args.sam_surrounding_decay, args.sam_boundary_width,
        )
        save_map_uint16(args.output_dir / "maps" / f"{sid}.png", fused)
        if args.visualize_category:
            save_visualization(visualize_dir / f"{sid}.png", image,
                               patch_pred.anomaly_map, sam_mask, fused, miou)
        score = float(fused.max()); predictions.append((sid, score))
        print(f"[predict] {sid} ({category}) patch={patch_pred.image_score:.5f} sam_mIoU={miou:.5f} score={score:.5f}", flush=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "image_score"])
        w.writerows((sid, f"{score:.6f}") for sid, score in predictions)
    print(f"[predict] done -> {args.output_dir} ({len(predictions)} samples)", flush=True)


if __name__ == "__main__": main()
