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
from torch.nn import functional as F
from torchvision import models as tv_models


def resolve_submodule(model: nn.Module, dotted: str) -> nn.Module:
    """按点分路径取子模块，数字段走下标（ModuleList 不能按 attribute 访问）。"""
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def _is_torchvision_vit(model: nn.Module) -> bool:
    """判断是否为 torchvision 原生 ViT（含固定 224 输入断言 + 可插值位置编码）。"""
    return (
        hasattr(model, "image_size")
        and hasattr(model, "patch_size")
        and hasattr(model, "conv_proj")
        and hasattr(model, "encoder")
        and hasattr(model.encoder, "pos_embedding")
    )


class _ViTFlexForward:
    """torchvision ViT 的「任意分辨率」前向包装。

    torchvision 的 ViT 硬编码 ``image_size=224``（``_process_input`` 里
    ``torch._assert(h == 224)``），且位置编码 ``encoder.pos_embedding`` 固定为
    ``(1, 1+14², D)``，无法直接吃 512 等其它分辨率。

    本包装绕过 224 断言，并把 14×14 位置编码网格**双线性插值**到实际输入网格
    （h/p × w/p），其余前向与原实现完全一致。**不改动 pos_embedding 的持久形状**，
    因此 shared.pth 仍存原始 (1, 197, D)，predict 端加载不 size mismatch。
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.patch = int(model.patch_size)
        # 原始位置编码 (1, 1+g², D)：首 token 为 CLS，其余为 g×g 网格
        pe = model.encoder.pos_embedding
        self._cls = pe[:, :1]              # (1, 1, D)
        grid = pe[:, 1:]                   # (1, g², D)
        g = int(math.isqrt(grid.shape[1]))
        d = pe.shape[-1]
        self._grid = grid.transpose(1, 2).reshape(1, d, g, g)  # (1, D, g, g)
        self._g = g

    def _pos_for(self, h: int, w: int) -> Tensor:
        """按输入网格 (h/p × w/p) 插值位置编码，返回 (1, 1+h·w, D)。"""
        gh, gw = h // self.patch, w // self.patch
        if (gh, gw) == (self._g, self._g):
            grid = self._grid
        else:
            grid = F.interpolate(self._grid, size=(gh, gw), mode="bilinear", align_corners=False)
        return torch.cat([self._cls, grid.reshape(1, self.model.encoder.pos_embedding.shape[-1], gh * gw).transpose(1, 2)], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """复制 torchvision ViT.forward，但用插值后的位置编码。

        不调用 encoder.forward（其内部 ``input + self.pos_embedding`` 固定 197 长度，
        与任意分辨率不兼容），改为等价实现 ``ln(layers(dropout(input)))``——
        layers 仍是原 Sequential，blocks.2/3 钩子照常触发，pos_embedding 不被改动。
        """
        model = self.model
        enc = model.encoder
        n = x.shape[0]
        x = model.conv_proj(x)  # (n, D, h/p, w/p)
        nh, nw = x.shape[-2:]
        x = x.reshape(n, model.hidden_dim, nh * nw).permute(0, 2, 1)
        x = torch.cat([model.class_token.expand(n, -1, -1), x], dim=1)
        x = x + self._pos_for(nh * model.patch_size, nw * model.patch_size)
        x = enc.dropout(x)
        x = enc.layers(x)   # hook 在此捕获 blocks.2 / blocks.3 特征
        x = enc.ln(x)
        return x[:, 0]  # 对齐原 forward 输出 CLS


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
        # ViT：用任意分辨率包装替代原 forward（CNN 主干走原路径）
        if _is_torchvision_vit(self.model):
            self._vit_flex = _ViTFlexForward(self.model)
        else:
            self._vit_flex = None
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
            if self._vit_flex is not None:
                self._vit_flex.forward(x)
            else:
                self.model(x)
        return self.features

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state(self, state: dict) -> None:
        """加载 state_dict（兼容 ``{"state_dict": ...}`` 包裹格式）。"""
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
