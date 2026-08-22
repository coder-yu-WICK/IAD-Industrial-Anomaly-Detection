# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 轻量实现（torch 原生，无 anomalib 依赖）。

对齐 anomalib Patchcore 技术路线：
    任意 torchvision 主干（默认 wide_resnet50_2，亦可 vit_base_patch16_224 等）
    → 指定层特征（默认 layer2+layer3，亦可 blocks.10/blocks.11）
    → 全量 patch coreset（JL 投影 + 贪心 k-center）
    → 最近邻 L2 打分 → 高斯平滑热图 → [0,1] 分数（可选 F1 阈值二值输出）。
"""

from .anomaly_map import AnomalyMapGenerator
from .backbone import PatchBackbone
from .coreset import CoresetSampler, SparseRandomProjection
from .features import PatchFeatureExtractor
from .memory_bank import MemoryBank
from .manifest import write_manifest
from .model import PatchCore, Prediction
from .onnx import OnnxBackbone
from .preprocess import PatchPreprocess
from .threshold import F1AdaptiveThreshold

__all__ = [
    "AnomalyMapGenerator",
    "CoresetSampler",
    "F1AdaptiveThreshold",
    "MemoryBank",
    "OnnxBackbone",
    "PatchBackbone",
    "PatchCore",
    "PatchFeatureExtractor",
    "PatchPreprocess",
    "Prediction",
    "SparseRandomProjection",
    "write_manifest",
]
