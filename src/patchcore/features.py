# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""多尺度 patch 特征：layer2+layer3 对齐拼接 + 3×3 局部平均池化。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import PatchBackbone


class PatchFeatureExtractor:
    """对单图提取 patch 特征 (C, h, w)。

    流程同 PatchCore：layer3 上采样到 layer2 分辨率后按通道拼接，
    再做 3×3 平均池化（neighborhood aggregation）。
    """

    def __init__(self, backbone: PatchBackbone, layers=("layer2", "layer3")) -> None:
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, image: Tensor) -> Tensor:
        """输入 (3, H, W) → 输出 (C, h, w)。"""
        features = self.backbone.forward(image.unsqueeze(0))
        ref = features[self.layers[0]]
        pooled = []
        for name in self.layers:
            f = features[name]
            if f.shape[-2:] != ref.shape[-2:]:
                f = F.interpolate(f, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(self.pool(f))
        return torch.cat(pooled, dim=1)[0]
