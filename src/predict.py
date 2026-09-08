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
    p.add_argument("--sam-model-type", choices=tuple(SAM_CHECKPOINT_URLS), default="vit_h")
    p.add_argument("--sam-points-per-side", type=int, default=32,
                   help="SAM 自动掩码采样密度，需与本地验证代码一致")
    p.add_argument("--sam-stability-score-thresh", type=float, default=0.92)
    p.add_argument("--sam-iou-threshold", type=float, default=0.35,
                   help="兼容旧参数，不再用于筛选 SAM mask")
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
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.sam_points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=args.sam_stability_score_thresh,
    )


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


def fuse_map(patch_map: np.ndarray, sam_masks: list[np.ndarray],
             surrounding_decay: float, boundary_width: int = 5,
             min_gradient: float = 0.01) -> np.ndarray:
    """用所有 SAM 轮廓重排热力图空间分布，但严格守恒原始分数。

    这里不能直接给像素加减分数：那会改变热力图的直方图和均值。先用
    SAM mask 构造“应该排在前面”的空间顺序，再把原 PatchCore 分数按
    rank 一一映射回去。因此输出与输入拥有完全相同的分数集合、均值和
    最大/最小值，只改变高低分的空间位置。低对比度的错误 mask 权重接近 0。
    """
    patch = np.clip(np.asarray(patch_map, dtype=np.float32), 0.0, 1.0)
    if not sam_masks:
        return patch.copy()
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except ImportError:
        return patch.copy()

    h, w = patch.shape
    width = max(1, int(boundary_width))
    strength = float(np.clip(surrounding_decay, 0.0, 1.0))
    # target 只用于排序，不作为输出值。
    target = patch.astype(np.float32).copy()
    global_std = float(patch.std()) + 1e-6
    for raw_mask in sam_masks:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != patch.shape or not mask.any() or mask.all():
            continue
        inside = distance_transform_edt(mask)
        outside = distance_transform_edt(~mask)
        signed = inside - outside
        boundary = np.abs(signed) <= width
        if not boundary.any():
            continue
        # 只让 PatchCore 已经有区分度的 mask 参与，均匀低分网格不会主导结果。
        ring = (np.abs(signed) > width) & (np.abs(signed) <= 3 * width)
        inside_mean = float(patch[mask].mean())
        ring_mean = float(patch[ring].mean()) if ring.any() else float(patch.mean())
        contrast = max(0.0, inside_mean - ring_mean) / global_std
        support = float(np.clip(contrast, 0.0, 1.0))
        if support <= 1e-4:
            continue

        # mask 内部是形状先验，边界窄带是等值线先验；两者只影响排序。
        soft_shape = gaussian_filter(mask.astype(np.float32), sigma=max(0.5, width / 2.0))
        contour = np.exp(-np.abs(signed) / float(width)).astype(np.float32)
        contour[~boundary] = 0.0
        shape_prior = 0.7 * soft_shape + 0.3 * contour
        target += strength * support * shape_prior

    # 稳定排序后做 rank-preserving remap：每个原始分数恰好使用一次。
    source_values = np.sort(patch.reshape(-1), kind="stable")
    order = np.argsort(target.reshape(-1), kind="stable")
    fused_flat = np.empty_like(source_values)
    fused_flat[order] = source_values
    fused = fused_flat.reshape(h, w)
    return fused.astype(np.float32)


def save_map_uint16(path: Path, anomaly_map: np.ndarray) -> None:
    u16 = np.clip(np.round(np.asarray(anomaly_map) * 65535), 0, 65535).astype(np.uint16)
    Image.fromarray(u16).save(path)


def save_visualization(path: Path, image: Image.Image, patch_map: np.ndarray,
                       sam_mask: np.ndarray, fused_map: np.ndarray, miou: float,
                       sam_masks: list[np.ndarray] | None = None) -> None:
    """保存四联图，并逐条绘制所有 SAM 轮廓（不只显示并集）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    rgb = np.asarray(image)
    axes[0].imshow(rgb); axes[0].set_title("Original")
    axes[1].imshow(rgb); axes[1].imshow(patch_map, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("PatchCore heatmap")
    axes[2].imshow(rgb)
    if sam_masks:
        # 画所有 mask 的轮廓，避免并集视觉上看起来只有一两块。
        for mask in sam_masks:
            axes[2].contour(mask.astype(np.float32), levels=[0.5], colors="lime", linewidths=0.7)
    else:
        axes[2].imshow(sam_mask, cmap="spring", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(f"SAM contours ({len(sam_masks or [])}, mIoU={miou:.3f})")
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
                sam_masks = [np.asarray(item["segmentation"], dtype=bool) for item in sam_items]
                # 仅用于日志，不参与筛选；所有 SAM mask 都会进入边界融合。
                patch_binary = patch_pred.anomaly_map >= args.patch_threshold
                ious = []
                for mask in sam_masks:
                    union = np.logical_or(mask, patch_binary).sum()
                    ious.append(float(np.logical_and(mask, patch_binary).sum() / union) if union else 0.0)
                sam_mask = np.logical_or.reduce(sam_masks) if sam_masks else np.zeros(patch_binary.shape, bool)
                miou = float(np.mean(ious)) if ious else 0.0
            else:
                sam_masks, sam_mask, miou = [], np.zeros(patch_pred.anomaly_map.shape, bool), 0.0
        print(f"[predict] {sid}: SAM masks={len(sam_masks)}", flush=True)
        fused = fuse_map(
            patch_pred.anomaly_map, sam_masks, args.sam_surrounding_decay,
            args.sam_boundary_width,
        )
        save_map_uint16(args.output_dir / "maps" / f"{sid}.png", fused)
        if args.visualize_category:
            save_visualization(visualize_dir / f"{sid}.png", image,
                               patch_pred.anomaly_map, sam_mask, fused, miou, sam_masks)
        score = float(fused.max()); predictions.append((sid, score))
        print(f"[predict] {sid} ({category}) patch={patch_pred.image_score:.5f} sam_mIoU={miou:.5f} score={score:.5f}", flush=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "image_score"])
        w.writerows((sid, f"{score:.6f}") for sid, score in predictions)
    print(f"[predict] done -> {args.output_dir} ({len(predictions)} samples)", flush=True)


if __name__ == "__main__": main()
