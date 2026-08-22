# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""多尺度 patch 特征：逐层特征对齐 + 3×3 局部平均池化。

级联检索需要各层**独立**的特征（不拼接），故 ``__call__`` 返回
``{layer: (C, h, w)}``；``concat_feature`` 保留旧拼接行为，供 legacy
（单 bank）bank_dict 的推理路径使用。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import PatchBackbone


class PatchFeatureExtractor:
    """对单图提取逐层 patch 特征 dict；各层对齐到首层分辨率并 3×3 平均池化。

    注：ViT 全深度 token 网格恒定（如 518 输入恒为 37×37），层间分辨率天然一致，
    插值分支实际不触发；CNN 等变分辨率主干仍兼容。
    """

    def __init__(self, backbone: PatchBackbone, layers=("blocks.3", "blocks.6", "blocks.9")) -> None:
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, image: Tensor) -> dict[str, Tensor]:
        """输入 (3, H, W) → 输出 {layer: (C, h, w)}。"""
        features = self.backbone.forward(image.unsqueeze(0))
        ref = features[self.layers[0]]
        out: dict[str, Tensor] = {}
        for name in self.layers:
            f = features[name]
            if f.shape[-2:] != ref.shape[-2:]:
                f = F.interpolate(f, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            out[name] = self.pool(f)[0]
        return out

    def concat_feature(self, image: Tensor) -> Tensor:
        """按 self.layers 顺序拼接各层 → (ΣC, h, w)（legacy 单 bank 用）。"""
        feats = self(image)
        return torch.cat([feats[name] for name in self.layers], dim=0)
