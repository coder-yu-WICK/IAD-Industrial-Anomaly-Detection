# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主干：动态构造 torchvision 模型 → 截断到最深层 → forward 返回特征 → ONNX 导出。

主干名与层路径均使用 torchvision 原生命名，如：
    name="wide_resnet50_2",  layers=("layer2", "layer3")                 # CNN
    name="vit_b_16",         layers=("encoder.layers.2", "encoder.layers.3")  # ViT
    name="swin_t",           layers=("features.3", "features.5")          # Swin（28×28 + 14×14）

关键优化（相比旧版钩子捕获）：
    1. **截断**：只保留「最深层特征层」之前的子模块，删除其后所有深层
       （ResNet 删 layer4/avgpool/fc；ViT 删 encoder.layers[k+1:]/ln/heads）。
       forward 不再走完整推理，state_dict 也不再包含深层权重。
    2. **forward 返回特征**（而非钩子旁路），可直接被 ``torch.onnx.export``
       追踪，用于 ONNX 加速推理。

ViT 输出为 3D token 序列，forward 内自动重排为 2D 方形网格（含 CLS 时丢弃首 token）。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision import models as tv_models


# ---------------------------------------------------------------------------
# 子模块解析
# ---------------------------------------------------------------------------

def resolve_submodule(model: nn.Module, dotted: str) -> nn.Module:
    """按点分路径取子模块，数字段走下标（Sequential/ModuleList 不能按 attribute 访问）。"""
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


# ---------------------------------------------------------------------------
# token 序列 → 2D 网格
# ---------------------------------------------------------------------------

def to_spatial(output: Tensor, num_register_tokens: int = 0) -> Tensor:
    """3D token 序列 (N, L, D) → 4D 特征图 (N, D, h, w)；已是 4D 则原样返回。

    含额外 token（L = 1 + r + h²，cls + r 个寄存器 token）时丢弃前置 token；
    无额外 token（L = h²）时保留全部。
    """
    if output.dim() != 3:
        return output
    n, l, d = output.shape
    g = int(math.sqrt(l - 1 - num_register_tokens))
    if g * g == l - 1 - num_register_tokens:
        tokens, grid = output[:, 1 + num_register_tokens:, :], g
    elif g * g == l:
        tokens, grid = output, g
    else:
        raise ValueError(f"无法将 {l} 个 token 重排为方形 grid（需 L=h² 或 1+r+h²）")
    return tokens.permute(0, 2, 1).reshape(n, d, grid, grid)


# ---------------------------------------------------------------------------
# 截断包装（架构族专用）
# ---------------------------------------------------------------------------

_RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")


class _TruncatedResNet(nn.Module):
    """ResNet 截断：走到最深层特征层即返回被钩层特征，深层权重不注册。

    state_dict 键名与原 torchvision 一致（``conv1.*`` / ``layerN.*``），
    仅缺深层（layer4/avgpool/fc）。
    """

    def __init__(self, resnet: nn.Module, layers: tuple[str, ...]) -> None:
        super().__init__()
        self.layers = tuple(layers)
        deepest = max(_RESNET_STAGES.index(l) for l in self.layers)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        for s in _RESNET_STAGES[: deepest + 1]:
            setattr(self, s, getattr(resnet, s))
        self._stage_names = list(_RESNET_STAGES[: deepest + 1])

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs: dict[str, Tensor] = {}
        for s in self._stage_names:
            x = getattr(self, s)(x)
            if s in self.layers:
                outs[s] = x
        return tuple(outs[l] for l in self.layers)


class _TruncatedViT(nn.Module):
    """torchvision ViT 截断：patch embed → blocks[0..k] → 返回被钩层 2D 特征。

    复现 torchvision ``VisionTransformer.forward`` 的前半段（conv_proj → reshape
    → cat cls → +pos_embed → dropout → 逐 block），在最深层 block 后停手；
    ``encoder.ln`` 与 ``heads`` 不注册，权重与计算一并裁剪。
    """

    def __init__(self, vit: nn.Module, layers: tuple[str, ...], num_register_tokens: int = 0) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.num_register_tokens = num_register_tokens
        self._deepest = max(int(l.rsplit(".", 1)[-1]) for l in self.layers)

        # 输入处理子模块（复用原模块，键名与 torchvision 对齐）
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.hidden_dim = vit.hidden_dim
        self.patch_size = vit.patch_size
        self.encoder = nn.Module()
        self.encoder.pos_embedding = vit.encoder.pos_embedding
        self.encoder.dropout = vit.encoder.dropout
        # 只保留 blocks[0..k]；键名用数字下标（save/load 内部自洽，预训练在截断前加载）
        self.encoder.layers = nn.Sequential(
            OrderedDict((str(i), vit.encoder.layers[i]) for i in range(self._deepest + 1))
        )

    def _process_input(self, x: Tensor) -> Tensor:
        n, _, h, w = x.shape
        p = self.patch_size
        x = self.conv_proj(x)  # (n, D, h/p, w/p)
        x = x.reshape(n, self.hidden_dim, (h // p) * (w // p))
        x = x.permute(0, 2, 1)  # (n, N, D)
        return torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = self._process_input(x)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        outs: dict[str, Tensor] = {}
        for i in range(self._deepest + 1):
            x = self.encoder.layers[i](x)
            name = f"encoder.layers.{i}"
            if name in self.layers:
                outs[name] = to_spatial(x, self.num_register_tokens)
        return tuple(outs[l] for l in self.layers)


class _TruncatedSwin(nn.Module):
    """torchvision Swin 截断：走到最深层 hook stage 即返回被钩层 2D 特征。

    Swin 的 ``features`` 是 8 段 Sequential（PatchEmbed → 4 个 stage 的 block /
    PatchMerging），逐段输出 channel-last (N, H, W, C) 特征图。截断到最深层
    stage，返回各 hook 层并统一 permute 到 channel-first (N, C, H, W)。
    深层（norm / head / 后续 stage）不注册，权重与计算一并裁剪。
    """

    def __init__(self, swin: nn.Module, layers: tuple[str, ...]) -> None:
        super().__init__()
        self.layers = tuple(layers)
        idxs = sorted({int(l.rsplit(".", 1)[-1]) for l in self.layers})
        self._deepest = max(idxs)
        self._hook_idxs = set(idxs)
        # 只保留 features[0..deepest]；键名与原 torchvision 一致（features.N.*）
        self.features = swin.features[: self._deepest + 1]

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outs: dict[str, Tensor] = {}
        for i in range(self._deepest + 1):
            x = self.features[i](x)  # (N, H, W, C)
            if i in self._hook_idxs:
                outs[f"features.{i}"] = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return tuple(outs[l] for l in self.layers)


def _truncate(model: nn.Module, layers: tuple[str, ...], num_register_tokens: int = 0) -> nn.Module:
    """按架构族分派截断包装。"""
    if all(hasattr(model, a) for a in ("layer4", "avgpool", "fc")):
        return _TruncatedResNet(model, layers)
    if hasattr(model, "encoder") and hasattr(model, "heads"):
        return _TruncatedViT(model, layers, num_register_tokens)
    if hasattr(model, "features") and hasattr(model, "head"):
        return _TruncatedSwin(model, layers)
    raise ValueError(f"不支持截断的主干（需 ResNet / torchvision ViT / Swin 结构）")


# ---------------------------------------------------------------------------
# 主干
# ---------------------------------------------------------------------------

class PatchBackbone(nn.Module):
    """冻结的预训练主干（截断到最深层），forward 返回逐层 4D 特征，可导出 ONNX。

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
        super().__init__()
        self.device = device
        self.name = name
        self.layers = tuple(layers)
        builder = getattr(tv_models, name, None)
        if not callable(builder):
            raise ValueError(f"torchvision.models 中没有可调用的主干 {name}")
        model = builder(weights=None)  # 不联网下载
        if pretrained_path is not None:
            self._load_state(model, torch.load(pretrained_path, map_location="cpu", weights_only=False))
        self.model = _truncate(model, self.layers, getattr(model, "num_register_tokens", 0))
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _load_state(model: nn.Module, state: dict) -> None:
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """输入 (N, 3, H, W) → 按 ``self.layers`` 顺序返回 (N, C, h, w) 特征。"""
        return self.model(x)

    def features(self, x: Tensor) -> dict[str, Tensor]:
        """同 forward，但返回 {layer: 特征} 字典（PyTorch 训练/推理路径用）。"""
        return dict(zip(self.layers, self.forward(x)))

    def state_dict(self, *args, **kwargs):
        """仅含截断后浅层权重（深层已从 self.model 移除）。"""
        return self.model.state_dict(*args, **kwargs)

    def load_state(self, state: dict) -> None:
        """加载截断后的 state_dict（兼容 ``{"state_dict": ...}`` 包裹格式）。"""
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------ ONNX
    def export_onnx(self, path: Path | str, input_size=(224, 224)) -> None:
        """把截断主干导出为 ONNX（输出为逐层特征，名称 = 层路径）。

        Args:
            path: 输出 .onnx 路径。
            input_size: 预处理后的输入 (H, W)，默认 (224, 224)（与 crop_size 一致）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, *input_size, dtype=torch.float32, device=self.device)
        output_names = [l.replace(".", "_") for l in self.layers]
        torch.onnx.export(
            self,
            dummy,
            str(path),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=None,  # 固定输入尺寸，避免动态形状降低推理速度
            opset_version=18,
            external_data=False,  # 权重内联为单个 .onnx 文件，便于拷贝与断网评测
        )
