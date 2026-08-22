# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 风格 ViT-B/14：加载 Franca 预训练主干（键名对齐 Franca 命名空间）。

Franca（CVPR 2026, arXiv:2507.14137）基于 DINOv2/iBOT 自监督框架。实测其
ViT-B/14 主干键名与标准 DINOv2 有三处不同（scripts/fetch_franca.py 已确认）：
  - MLP 为 **SwiGLU**（``blocks.N.mlp.w12/w3``），非标准 ``fc1/fc2``；
  - LayerScale 为带 ``gamma`` 参数的模块（``blocks.N.ls1.gamma``），非裸参数；
  - 顶层多一个 iBOT ``mask_token``（训练用，推理不影响，仅加载对齐）。

`blocks` 为 ``nn.ModuleList`` 属性，供 ``resolve_submodule("blocks.3")``
按数字下标访问做前向钩子；输出为 3D token 序列，由 backbone 钩子重排为 2D 网格。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """14×14 无重叠卷积 patch 嵌入，(B,C,H,W) → (B, h*w, D)。"""

    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, D, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, h*w, D)


class Attention(nn.Module):
    """多头自注意力，融合 qkv 线性层（DINOv2/Franca 键名 ``qkv``）。"""

    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * (self.head_dim**-0.5), qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.proj(x)


class LayerScale(nn.Module):
    """逐通道缩放（DINOv2/Franca 键名 ``gamma``）。"""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    """SwiGLU 前馈（Franca 键名 ``w12``/``w3``）。

    fused=False（默认，标准 SwiGLU）：``w12`` 输出 2×hidden，切两半后
    ``silu(前半)*后半`` 再经 ``w3`` 降维；fused=True 对应 xFormers fused 变体
    （对 ``w12`` 整体 ``silu`` 后 ``w3`` 降维）。变体与隐藏维由
    fetch_franca.py 按 ref 实际权重形状判定后传入，避免猜测。
    """

    def __init__(self, in_features: int = 768, hidden: int = 3072, fused: bool = False) -> None:
        super().__init__()
        self.fused = fused
        self.w12 = nn.Linear(in_features, 2 * hidden)
        self.w3 = nn.Linear(2 * hidden if fused else hidden, in_features)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        if self.fused:
            return self.w3(F.silu(x12))
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Block(nn.Module):
    """Transformer block：norm1→attn→ls1 残差，norm2→mlp→ls2 残差。"""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_hidden: int = 3072, mlp_fused: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden, mlp_fused)
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoViT(nn.Module):
    """DINOv2 风格 ViT-B/14（12 层、768 维、patch 14，可含寄存器 token）。"""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_hidden: int = 3072,
        mlp_fused: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        n_extra = 1 + num_register_tokens  # cls + 寄存器 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))  # iBOT 训练用，推理不参与
        if num_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + n_extra, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_hidden, mlp_fused) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """随机初始化（未加载权重时的兜底；正常流程由 Franca 权重覆盖）。"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, x: Tensor) -> Tensor:
        """把预训练 (1, 1+37²+r, D) 位置编码双三次插值到当前输入 grid。

        输入等于预训练分辨率（518→37×37）时原样返回，不改 pos_embed。
        """
        num_extra = 1 + self.num_register_tokens
        n_cur = x.shape[1]
        g = int(math.isqrt(self.pos_embed.shape[1] - num_extra))
        grid = int(math.isqrt(n_cur - num_extra))
        if grid == g:
            return self.pos_embed
        pe = self.pos_embed  # (1, 1+r+g², D)
        d = self.pos_embed.shape[-1]
        head, toks = pe[:, :num_extra], pe[:, num_extra:]
        toks = toks.transpose(1, 2).reshape(1, d, g, g)
        toks = F.interpolate(toks, size=(grid, grid), mode="bicubic", align_corners=False)
        return torch.cat([head, toks.reshape(1, d, grid * grid).transpose(1, 2)], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  # (B, N, D)
        b = x.shape[0]
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        if self.num_register_tokens:
            x = torch.cat([x, self.register_tokens.expand(b, -1, -1)], dim=1)
        x = x + self._interpolate_pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def build_franca_vitb14(
    img_size: int = 518,
    num_register_tokens: int = 0,
    mlp_hidden: int | None = None,
    mlp_fused: bool = False,
) -> DinoViT:
    """Franca ViT-B/14 工厂（供 backbone.py 自定义主干注册表调用）。

    ``mlp_hidden``/``mlp_fused`` 由 fetch_franca.py 按 ref 实际权重形状判定后
    传入；默认 3072 / chunked 为标准 SwiGLU（与 ViT-B 4× 扩张一致）。
    """
    return DinoViT(
        img_size=img_size,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_hidden=mlp_hidden if mlp_hidden is not None else 3072,
        mlp_fused=mlp_fused,
        num_register_tokens=num_register_tokens,
    )
