# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主干：按名称动态构造 torchvision 模型 + 任意层前向钩子 + 离线权重加载。

主干名与层路径均使用 torchvision 原生命名，如：
    name="wide_resnet50_2",  layers=("layer2", "layer3")        # CNN
    name="vit_b_16",         layers=("encoder.layers.10", "encoder.layers.11")  # ViT
ViT 输出为 3D token 序列，钩子里自动重排为 2D 方形网格（含 CLS 时丢弃首 token）。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision import models as tv_models


def resolve_submodule(model: nn.Module, dotted: str) -> nn.Module:
    """按点分路径取子模块，数字段走下标（ModuleList 不能按 attribute 访问）。"""
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


class PatchBackbone:
    """冻结的预训练主干，在给定层注册前向钩子，输出统一 4D 特征图。

    Args:
        device: 运行设备。
        pretrained_path: 预训练权重文件路径；为 None 时用随机初始化
            （断网评测环境由 predict 从 shared.pth 加载）。
        name: ``torchvision.models`` 中任意主干名，默认 ``wide_resnet50_2``。
        layers: 特征层路径序列，默认 ("layer2", "layer3")。
    """

    def __init__(
        self,
        device: torch.device,
        pretrained_path: Path | str | None = None,
        name: str = "wide_resnet50_2",
        layers=("layer2", "layer3"),
    ) -> None:
        self.device = device
        self.name = name
        self.layers = tuple(layers)
        builder = getattr(tv_models, name, None)
        if not callable(builder):
            raise ValueError(f"torchvision.models 中没有可调用的主干 {name}")
        self.model = builder(weights=None)  # 不联网下载
        if pretrained_path is not None:
            self.load_state(torch.load(pretrained_path, map_location="cpu", weights_only=False))
        self.model = self.model.to(device)
        self.model.eval()
        self.features: dict[str, Tensor] = {}
        for layer in self.layers:
            resolve_submodule(self.model, layer).register_forward_hook(self._make_hook(layer))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.features[name] = self._to_spatial(output)

        return hook

    @staticmethod
    def _to_spatial(output: Tensor) -> Tensor:
        """3D token 序列 (N, L, D) → 4D 特征图 (N, D, h, w)；已是 4D 则原样返回。

        含 CLS（L = h²+1）时丢弃首 token；无 CLS（L = h²）时保留全部。
        """
        if output.dim() != 3:
            return output
        n, l, d = output.shape
        g = int(math.sqrt(l - 1))
        if g * g == l - 1:  # 含 CLS token
            tokens, grid = output[:, 1:, :], g
        elif g * g == l:  # 无 CLS
            tokens, grid = output, g
        else:
            raise ValueError(f"无法将 {l} 个 token 重排为方形 grid（需 L=h² 或 h²+1）")
        return tokens.permute(0, 2, 1).reshape(n, d, grid, grid)

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
