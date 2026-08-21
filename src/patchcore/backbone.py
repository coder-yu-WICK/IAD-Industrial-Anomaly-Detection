# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主干：wide_resnet50_2 + layer2/layer3 前向钩子 + 离线权重加载。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.models import wide_resnet50_2


class PatchBackbone:
    """冻结的 ImageNet 预训练 wide_resnet50_2，输出 layer2/layer3 特征。

    Args:
        device: 运行设备。
        pretrained_path: 预训练权重文件路径；为 None 时用随机初始化
            （断网评测环境由 predict 从 shared.pth 加载）。
        name: 主干名称，仅支持 ``wide_resnet50_2``。
    """

    def __init__(
        self,
        device: torch.device,
        pretrained_path: Path | str | None = None,
        name: str = "wide_resnet50_2",
    ) -> None:
        if name != "wide_resnet50_2":
            raise ValueError(f"仅支持 wide_resnet50_2，收到 {name}")
        self.device = device
        self.model = wide_resnet50_2(weights=None)  # 不联网下载
        if pretrained_path is not None:
            self.load_state(torch.load(pretrained_path, map_location="cpu", weights_only=False))
        self.model = self.model.to(device)
        self.model.eval()
        self.features: dict[str, Tensor] = {}
        self.model.layer2.register_forward_hook(self._make_hook("layer2"))
        self.model.layer3.register_forward_hook(self._make_hook("layer3"))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.features[name] = output

        return hook

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """前向一次并返回 {layer: 特征}，输入 (1, 3, H, W)。"""
        self.features.clear()
        with torch.inference_mode():
            self.model(x)
        return self.features

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state(self, state: dict) -> None:
        """加载 state_dict（兼容 ``{"state_dict": ...}`` 包裹格式）。"""
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
