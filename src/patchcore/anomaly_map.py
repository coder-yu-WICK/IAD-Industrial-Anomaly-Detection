# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""异常热图生成：上采样到原图尺寸 + 高斯平滑（对齐 anomalib AnomalyMapGenerator）。"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as tvF


class AnomalyMapGenerator:
    """patch 分数图 → 原图尺寸热图，先双线性上采样、后高斯平滑。

    Args:
        sigma: 高斯核标准差，默认 4.0（kernel_size=33）。
    """

    def __init__(self, sigma: float = 4.0) -> None:
        self.sigma = float(sigma)
        self.kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1

    def __call__(self, patch_scores: Tensor, image_size: tuple[int, int]) -> Tensor:
        """patch_scores (1, h, w) → (1, H, W) 平滑热图。image_size 为 (H, W)。"""
        up = F.interpolate(
            patch_scores.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, H, W)
        up = tvF.gaussian_blur(up, kernel_size=self.kernel_size, sigma=self.sigma)
        return up[0]  # (1, H, W)
