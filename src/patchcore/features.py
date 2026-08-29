# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""多尺度 patch 特征：多个特征层对齐拼接 + 3×3 局部平均池化。

``__call__`` 走 PyTorch 主干；``aggregate`` 只做对齐 + 池化 + 拼接，
供 ONNX 路径复用（ONNX 主干输出逐层特征后，聚合逻辑与此完全一致）。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import PatchBackbone


def resolve_inference_dtype(device_type: str, dtype: str | None) -> str:
    """把 None/"float32"/"float16"/"bfloat16" 规约为实际 dtype 名；非 CUDA 恒为 float32。

    None（默认）→ CUDA 用 float16、CPU 用 float32；显式 float32/fp32/none → 禁用半精度。
    """
    if dtype is None:
        dtype = "float16" if device_type == "cuda" else "float32"
    if device_type != "cuda" or dtype in ("float32", "fp32", "none"):
        return "float32"
    if dtype in ("float16", "fp16", "half"):
        return "float16"
    if dtype in ("bfloat16", "bf16"):
        return "bfloat16"
    return "float16"


class PatchFeatureExtractor:
    """对单图提取 patch 特征 (C, h, w)。

    流程同 PatchCore：各层上采样到第一层分辨率后按通道拼接，
    再做 3×3 平均池化（neighborhood aggregation）。
    """

    def __init__(
        self,
        backbone: PatchBackbone,
        layers=("layer2", "layer3"),
        inference_dtype: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.layers = tuple(layers)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        # None → CUDA 默认 FP16 推理；matmul 半精度走 tensor core，残差仍 FP32，
        # 最终特征以 FP32 输出（与 FP32 bank 一致），指标基本无损。显式 float32 可禁用对照。
        self.inference_dtype = inference_dtype

    def _autocast_ctx(self):
        dtype = resolve_inference_dtype(self.backbone.device.type, self.inference_dtype)
        if dtype == "float32":
            return nullcontext()
        t = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.autocast(self.backbone.device.type, dtype=t)

    def __call__(self, image: Tensor) -> Tensor:
        """输入 (3, H, W) → 输出 (C, h, w)（PyTorch 主干路径）。

        特征提取为纯推理，用 inference_mode 关闭 autograd（与冻结主干语义一致，
        也避免下游 ``.numpy()`` 因 requires_grad 报错）；CUDA 下套 autocast 加速主干 GEMM。
        """
        with torch.inference_mode(), self._autocast_ctx():
            named = self.backbone.features(image.unsqueeze(0))
        return self.aggregate(named)

    def aggregate(self, features: dict[str, Tensor]) -> Tensor:
        """输入 {layer: (1, C, h, w)} → 对齐 + 3×3 池化 + 拼接 → (ΣC, h, w)。"""
        ref = features[self.layers[0]]
        pooled = []
        for name in self.layers:
            f = features[name]
            if f.shape[-2:] != ref.shape[-2:]:
                f = F.interpolate(f, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(self.pool(f))
        return torch.cat(pooled, dim=1)[0]
