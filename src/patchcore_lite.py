# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 轻量实现（torch 原生，无 anomalib 依赖）。

对齐 anomalib Patchcore 的技术路线：
    backbone = wide_resnet50_2 (ImageNet 预训练，冻结)
    特征层   = layer2 + layer3（拼接后做局部平均池化）
    memory bank = 正常样本 patch 特征 + 贪心 coreset 采样
    打分     = 测试 patch 特征到 bank 的最近邻 L2 距离

训练（train.py）产出：
    <model-dir>/shared.pth                # 主干 state_dict（离线加载必需）
    <model-dir>/checkpoints/<category>.pth # 每类 bank + coreset 索引 + 归一化参数
    <model-dir>/model_manifest.json        # 模型清单（hybrid 模式）

预测（predict.py）产出：
    像素级热图 = patch 距离图双线性上采样到原图，类别级归一化到 [0,1]
    图像级分数 = 热图最大值
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def build_backbone(device: torch.device, pretrained_path: Path | str | None = None):
    """构建 wide_resnet50_2 主干并注册前向钩子，返回 (model, features)。

    Args:
        pretrained_path: 预训练权重文件路径。为 None 时返回随机初始化模型
            （评测环境断网时由调用方从随包权重加载，见 predict.py）。
    """
    model = wide_resnet50_2(weights=None)  # 不联网下载
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    features: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(module, _input, output):
            features[name] = output

        return hook

    model.layer2.register_forward_hook(make_hook("layer2"))
    model.layer3.register_forward_hook(make_hook("layer3"))
    return model, features


def extract_patch_features(
    model: torch.nn.Module,
    features: dict[str, torch.Tensor],
    image: torch.Tensor,
    layers: tuple[str, ...] = ("layer2", "layer3"),
) -> torch.Tensor:
    """对单张图 (3,H,W) 提取多尺度 patch 特征，返回 (C, h, w)。

    流程同 PatchCore：layer3 上采样到 layer2 分辨率后按通道拼接，
    再做 3x3 平均池化（neighborhood aggregation）。
    """
    with torch.inference_mode():
        model(image.unsqueeze(0))

    feats = []
    ref = features[layers[0]]  # (1, C, h, w)
    for name in layers:
        f = features[name]
        if f.shape[-2:] != ref.shape[-2:]:
            f = F.interpolate(f, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(f)
    concat = torch.cat(feats, dim=1)  # (1, C_sum, h, w)
    concat = F.avg_pool2d(concat, kernel_size=3, stride=1, padding=1)
    return concat[0]  # (C_sum, h, w)


# ---------------------------------------------------------------------------
# Memory bank 构建（训练）
# ---------------------------------------------------------------------------

def build_bank(
    model: torch.nn.Module,
    features: dict[str, torch.Tensor],
    image_paths: list[Path],
    device: torch.device,
    coreset_ratio: float = 0.1,
    target_size: tuple[int, int] | None = (224, 224),
) -> dict:
    """从正常样本构建类别 memory bank，返回可序列化的 dict。

    包含：bank 特征、coreset 索引、类别归一化参数（训练集最大 patch 距离）。
    """
    from PIL import Image
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size) if target_size else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    if target_size is None:
        preprocess = transforms.ToTensor()

    all_features: list[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img).to(device)
        all_features.append(extract_patch_features(model, features, x))

    # (N_patches, C)
    all_feats = torch.cat([f.reshape(f.shape[0], -1).T for f in all_features], dim=0)

    # 控制全量规模：过多 patch 先均匀随机子采样（保证 coreset 可训练）
    max_embed = 10000
    if all_feats.shape[0] > max_embed:
        perm = torch.randperm(all_feats.shape[0])[:max_embed]
        all_feats = all_feats[perm]

    if coreset_ratio is not None and coreset_ratio < 1.0:
        idx = greedy_coreset(all_feats, coreset_ratio)
        bank = all_feats[idx]
    else:
        idx = None
        bank = all_feats

    # 归一化 scale：全量训练 patch 到 bank 最近邻距离的均值（分块算，防爆内存）
    dists = []
    for i in range(0, all_feats.shape[0], 256):
        d = torch.cdist(all_feats[i : i + 256], bank)
        dists.append(d.min(dim=1).values)
    norm_scale = float(torch.cat(dists).mean())
    norm_scale = max(norm_scale, 1e-6)
    return {
        "bank": bank.cpu().numpy().astype(np.float32),
        "coreset_indices": None if idx is None else idx.cpu().numpy().tolist(),
        "norm_scale": norm_scale,
        "feature_dim": int(bank.shape[1]),
        "target_size": list(target_size) if target_size else None,
    }


def greedy_coreset(features: torch.Tensor, ratio: float) -> torch.Tensor:
    """贪心 minimax coreset 采样：每次选取离已选集最远的点，返回索引。"""
    n = features.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return torch.arange(n)

    indices = torch.zeros(k, dtype=torch.long, device=features.device)
    min_dist = torch.full((n,), float("inf"), device=features.device)
    current = int(torch.randint(0, n, (1,)).item())  # 随机起点（seed 由调用方保证）
    for i in range(k):
        indices[i] = current
        d = torch.norm(features - features[current], dim=1)
        min_dist = torch.minimum(min_dist, d)
        # 排除已选点
        min_dist[current] = -1.0
        current = int(torch.argmax(min_dist).item())
    return indices


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def predict_image(
    model: torch.nn.Module,
    features: dict[str, torch.Tensor],
    image: torch.Tensor,  # (3, H, W) 原图
    bank_dict: dict,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """单图推理，返回 (image_score ∈ [0,1], anomaly_map ∈ [0,1] 同原图尺寸)。"""
    bank = torch.from_numpy(np.asarray(bank_dict["bank"])).to(device)
    norm_scale = float(bank_dict.get("norm_scale") or 1.0)
    target_size = bank_dict.get("target_size")

    x = image.unsqueeze(0)
    if target_size is not None:
        x = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
    patch_feats = extract_patch_features(model, features, x[0])  # (C, h, w)

    h, w = patch_feats.shape[-2:]
    q = patch_feats.reshape(patch_feats.shape[0], -1).T  # (h*w, C)

    # 最近邻距离：GPU 分块 cdist，控制距离矩阵峰值显存
    chunk = 1024
    dist_chunks = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i : i + chunk], bank)  # (chunk, n_b)
        dist_chunks.append(d.min(dim=1).values)
    dist = torch.cat(dist_chunks)

    score_map = dist.reshape(1, h, w)

    # 上采样回原图尺寸
    if score_map.shape[-2:] != image.shape[-2:]:
        score_map = F.interpolate(
            score_map.unsqueeze(0), size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0]
    score_map = score_map[0]  # (H, W)

    # soft 单调压缩到 [0,1]：保序（AP/AUROC/F1-max 只依赖排序），且不饱和
    score_map = 1.0 - torch.exp(-score_map / norm_scale)
    image_score = float(score_map.max().item())
    return image_score, score_map.cpu().numpy()


# ---------------------------------------------------------------------------
# checkpoint 读写
# ---------------------------------------------------------------------------

def save_category_ckpt(path: Path, bank_dict: dict, category: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "category": category,
            "bank_dict": bank_dict,
        },
        path,
    )


def load_category_ckpt(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["bank_dict"]


def save_manifest(model_dir: Path, categories: list[str], model_mode: str = "hybrid") -> None:
    manifest = {
        "format_version": "omniad-school-model-1.0",
        "model_mode": model_mode,
        "checkpoint": "shared.pth",
        "checkpoint_pattern": "checkpoints/{category}.pth",
        "categories": categories,
        "score_range": [0.0, 1.0],
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
