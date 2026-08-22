# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 轻量实现（torch 原生，无 anomalib 依赖）。

技术路线：
    Franca 主干（DINOv2 风格 ViT-B/14，可换 torchvision 主干兼容旧 ckpt）
    → 浅/中/深多层特征（默认 blocks.3/6/9）
    → 每层独立 coreset bank（JL 投影 + 贪心 k-center，索引跨层对齐）
    → 级联 top-k 剪枝检索（浅层全量→中层子集→深层 min）
    → 高斯平滑热图 → [0,1] 分数（可选 F1 阈值二值输出）。
"""

from .anomaly_map import AnomalyMapGenerator
from .backbone import PatchBackbone
from .cascade import CascadeMemoryBank
from .coreset import CoresetSampler, SparseRandomProjection
from .features import PatchFeatureExtractor
from .manifest import write_manifest
from .memory_bank import MemoryBank
from .model import PatchCore, Prediction
from .preprocess import PatchPreprocess
from .threshold import F1AdaptiveThreshold
from .vit import DinoViT, build_franca_vitb14

__all__ = [
    "AnomalyMapGenerator",
    "CascadeMemoryBank",
    "CoresetSampler",
    "DinoViT",
    "F1AdaptiveThreshold",
    "MemoryBank",
    "PatchBackbone",
    "PatchCore",
    "PatchFeatureExtractor",
    "PatchPreprocess",
    "Prediction",
    "SparseRandomProjection",
    "build_franca_vitb14",
    "write_manifest",
]
