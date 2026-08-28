# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主干：torchvision 或自定义主干 + 任意层前向钩子 + 离线权重加载。

主干名/层路径使用模型自身命名空间，如：
    name="franca_vitb14",     layers=("blocks.3", "blocks.6", "blocks.9")  # 自定义 DINOv2 风格 ViT
    name="wide_resnet50_2",   layers=("layer2", "layer3")                 # torchvision CNN
    name="vit_b_16",          layers=("encoder.layers.10", "encoder.layers.11")  # torchvision ViT
ViT 输出为 3D token 序列，钩子里自动重排为 2D 方形网格（含 CLS/寄存器 token 时丢弃）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torchvision import models as tv_models

from .vit import build_franca_vitb14, infer_swiglu_from_state

# 自定义主干注册表：包内自实现、离线可用；未命中再走 torchvision 回退路径
# （torchvision 路径仅用于加载旧 ckpt 的兼容，非新方案回退）。
_CUSTOM_BACKBONES: dict[str, Callable[[], nn.Module]] = {
    "franca_vitb14": build_franca_vitb14,
}


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
        name: 主干名；自定义主干（``franca_vitb14``，默认）走注册表，
            torchvision 原生名仅用于加载旧 ckpt 的兼容。
        layers: 特征层路径序列，默认 ("blocks.3", "blocks.6", "blocks.9")。
    """

    def __init__(
        self,
        device: torch.device,
        pretrained_path: Path | str | None = None,
        name: str = "franca_vitb14",
        layers=("blocks.3", "blocks.6", "blocks.9"),
    ) -> None:
        self.device = device
        self.name = name
        self.layers = tuple(layers)
        # Franca 主干 SwiGLU 的隐藏维/变体依 checkpoint 实际权重形状判定
        # （fetch_franca.py 同规则；默认 3072 仅作随机初始化兜底）
        state = None
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
        self._build_model(**(infer_swiglu_from_state(state) or {}))
        if state is not None:
            self.model.load_state_dict(state)

    def _build_model(self, **builder_kwargs) -> None:
        """构建主干并注册前向钩子（重建时同样调用）。

        builder_kwargs 透传给自定义主干工厂（如 mlp_hidden/mlp_fused）；
        torchvision 路径仅用于加载旧 ckpt 的兼容，无建参。
        """
        builder = _CUSTOM_BACKBONES.get(self.name)
        if builder is not None:
            self.model = builder(**builder_kwargs)  # 自定义主干（不联网）
        else:
            builder = getattr(tv_models, self.name, None)
            if not callable(builder):
                raise ValueError(f"torchvision.models 中没有可调用的主干 {self.name}")
            self.model = builder(weights=None)  # 不联网下载
        # 钩子输入若为 3D token 序列，需知道几个前置 token（cls/寄存器）用于重排
        self.num_register_tokens = getattr(self.model, "num_register_tokens", 0)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.features: dict[str, Tensor] = {}
        for layer in self.layers:
            resolve_submodule(self.model, layer).register_forward_hook(self._make_hook(layer))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.features[name] = self._to_spatial(output, self.num_register_tokens)

        return hook

    @staticmethod
    def _to_spatial(output: Tensor, num_register_tokens: int = 0) -> Tensor:
        """3D token 序列 (N, L, D) → 4D 特征图 (N, D, h, w)；已是 4D 则原样返回。

        含额外 token（L = 1+r+h²，cls + r 个寄存器 token）时丢弃前置 token；
        无额外 token（L = h²）时保留全部。
        """
        if output.dim() != 3:
            return output
        n, l, d = output.shape
        g = int(math.sqrt(l - 1 - num_register_tokens))
        if g * g == l - 1 - num_register_tokens:
            tokens, grid = output[:, 1 + num_register_tokens :, :], g
        elif g * g == l:
            tokens, grid = output, g
        else:
            raise ValueError(f"无法将 {l} 个 token 重排为方形 grid")
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
        """加载 state_dict（兼容 ``{"state_dict": ...}`` 包裹格式）。

        若 checkpoint 的 SwiGLU 配置与当前模型不一致（如 Franca 隐藏维 2048
        ≠ 默认 3072），先按 checkpoint 实际权重形状重建主干再加载，避免
        size mismatch（predict 端 load_shared 同样走此路径）。
        """
        if "state_dict" in state:
            state = state["state_dict"]
        cfg = infer_swiglu_from_state(state)
        if cfg is not None and self.name in _CUSTOM_BACKBONES:
            s_w12 = state["blocks.0.mlp.w12.weight"].shape
            s_w3 = state["blocks.0.mlp.w3.weight"].shape
            m_w12 = self.model.blocks[0].mlp.w12.weight.shape
            m_w3 = self.model.blocks[0].mlp.w3.weight.shape
            if (m_w12[0], m_w3[1]) != (s_w12[0], s_w3[1]):
                print(
                    f"[backbone] SwiGLU 配置不匹配，按 checkpoint 重建主干: "
                    f"mlp_hidden={cfg['mlp_hidden']} mlp_fused={cfg['mlp_fused']}",
                    flush=True,
                )
                self._build_model(**cfg)
        self.model.load_state_dict(state)
