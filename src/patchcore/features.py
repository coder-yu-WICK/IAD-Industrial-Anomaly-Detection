# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""多尺度 patch 特征：多个特征层对齐拼接 + 3×3 局部平均池化。

``__call__`` 走 PyTorch 主干；``aggregate`` 只做对齐 + 池化 + 拼接，
供 ONNX 路径复用（ONNX 主干输出逐层特征后，聚合逻辑与此完全一致）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import PatchBackbone


class PatchFeatureExtractor:
    """对单图提取 patch 特征 (C, h, w)。

    流程同 PatchCore：各层上采样到第一层分辨率后按通道拼接，
    再做 3×3 平均池化（neighborhood aggregation）。
    """

    def __init__(self, backbone: PatchBackbone, layers=("layer2", "layer3")) -> None:
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, image: Tensor) -> Tensor:
        """输入 (3, H, W) → 输出 (C, h, w)（PyTorch 主干路径）。

        特征提取为纯推理，用 inference_mode 关闭 autograd（与冻结主干语义一致，
        也避免下游 ``.numpy()`` 因 requires_grad 报错）。
        """
        with torch.inference_mode():
            named = self.backbone.features(image.unsqueeze(0))
        return self.aggregate(named)

    def aggregate(self, features: dict[str, Tensor]) -> Tensor:
        """输入 {layer: (1, C, h, w)} → 对齐 + 3×3 池化 + 拼接 + L2 归一化 → (ΣC, h, w)。

        每个 patch 向量沿通道维做 L2 归一化，把欧氏最近邻等价为余弦相似度——
        对 DINOv2 这类高维 Transformer 特征更稳定（不受向量模长影响）。
        """
        ref = features[self.layers[0]]
        pooled = []
        for name in self.layers:
            f = features[name]
            if f.shape[-2:] != ref.shape[-2:]:
                f = F.interpolate(f, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(self.pool(f))
        out = torch.cat(pooled, dim=1)[0]  # (ΣC, h, w)
        return F.normalize(out, p=2, dim=0)  # 每个 patch 向量单位化
