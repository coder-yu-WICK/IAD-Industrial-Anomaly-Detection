# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""图像预处理：Resize → CenterCrop → ToTensor → Normalize。

train 与 predict 共用同一套参数；参数写入 bank ckpt，predict 时重建，
保证推理与训练预处理完全一致（对齐 anomalib PatchCore 论文配置）。
"""

from __future__ import annotations

from PIL import Image
from torch import Tensor
from torchvision import transforms


class PatchPreprocess:
    """PatchCore 图像预处理（对齐论文：256 resize → 224 center crop → ImageNet 归一化）。

    Args:
        input_size: 缩放目标 (H, W)，默认 (256, 256)。
        crop_size: 中心裁剪目标 (H, W)，默认 (224, 224)。
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, input_size=(256, 256), crop_size=(224, 224)) -> None:
        self.input_size = tuple(input_size)
        self.crop_size = tuple(crop_size)
        self._compose = transforms.Compose(
            [
                transforms.Resize(self.input_size, antialias=True),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
            ]
        )

    def encode(self, image: Image.Image) -> Tensor:
        """单图编码 → (3, crop_h, crop_w) float32 归一化 tensor。"""
        return self._compose(image.convert("RGB"))

    def to_dict(self) -> dict:
        """序列化为 bank_dict 字段。"""
        return {"input_size": list(self.input_size), "crop_size": list(self.crop_size)}

    @classmethod
    def from_dict(cls, data: dict) -> "PatchPreprocess":
        """从 bank_dict 重建（predict 必须用它，而非构造参数，保证与训练一致）。"""
        input_size = data.get("input_size")
        crop_size = data.get("crop_size")
        # 旧 ckpt 兼容：只有 target_size 时退化为缩放=裁剪（等价旧 lite 行为）
        if input_size is None and crop_size is None and "target_size" in data:
            ts = data["target_size"]
            input_size, crop_size = ts, ts
        return cls(input_size=input_size or (256, 256), crop_size=crop_size or (224, 224))
