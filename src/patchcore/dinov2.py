# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""标准 DINOv2 ViT-L/14 主干（键名对齐 facebookresearch/dinov2 官方命名空间）。

与 franca 分支的 ``vit.py``（Franca 自监督 ViT-B：SwiGLU + LayerScale 模块）不同，
这里是 **官方 DINOv2** 的结构，用于把 swin_t 换成更强的自监督 ViT-L 反超 franca：

  - MLP 为标准 GELU（``blocks.N.mlp.fc1/fc2``），非 SwiGLU ``w12/w3``；
  - LayerScale 为带 ``gamma`` 参数的模块（``blocks.N.ls1.gamma``）；
  - attention 用 ``qkv/proj``（均带 bias），走 scaled_dot_product_attention（与官方一致、GPU 上更快）；
  - 默认 4 个寄存器 token（对应 ``dinov2_vitl14_reg`` 权重），顺序 [cls, registers, patches]；
  - pos_embed 形状 (1, 1+patch数, D)，**不含**寄存器位。

仅实现推理所需最小结构，键名与官方 ``dinov2_vitl14_reg`` 完全一致，
由 scripts/fetch_dinov2.py 做前向一致性校验后打包权重。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """14×14 无重叠卷积 patch 嵌入（官方键名 ``patch_embed.proj``）。"""

    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, D, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, h*w, D)


class Attention(nn.Module):
    """多头自注意力（官方键名 ``attn.qkv/proj``，qkv/proj 均带 bias）。

    走 ``scaled_dot_product_attention``：与官方 DINOv2 完全一致；GPU 上自动走
    flash / memory-efficient attention，518 分辨率下显著更快（满足 100ms 的关键）。
    """

    def __init__(self, dim: int = 1024, num_heads: int = 16, qkv_bias: bool = True, proj_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)  # eval 无 dropout / 无 causal，等价官方
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    """标准 GELU 前馈（官方键名 ``mlp.fc1/fc2``）。"""

    def __init__(self, in_features: int = 1024, hidden_features: int = 4096, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    """逐通道缩放（官方键名 ``gamma``）。"""

    def __init__(self, dim: int = 1024, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Transformer block：norm1→attn→ls1 残差，norm2→mlp→ls2 残差。"""

    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0, init_values: float = 1.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls1 = LayerScale(dim, init_values)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2(nn.Module):
    """标准 DINOv2 ViT（ViT-L/14：24 层、1024 维、16 头、4 寄存器 token）。"""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        init_values: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        if num_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        # 官方 pos_embed 只含 [cls, patches]，不含寄存器位（寄存器无位置编码）
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.num_register_tokens:
            nn.init.normal_(self.register_tokens, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, x: Tensor) -> Tensor:
        """把 (1, 1+g0², D) 位置编码插值到当前 grid（518 输入时原样返回）。

        固定输入 518→37×37，grid 不变直接返回；仅作非 518 输入的兜底。
        """
        grid = int(math.isqrt(x.shape[1] - 1))
        g0 = int(math.isqrt(self.pos_embed.shape[1] - 1))
        if grid == g0:
            return self.pos_embed
        d = self.pos_embed.shape[-1]
        head, toks = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        toks = toks.transpose(1, 2).reshape(1, d, g0, g0)
        toks = F.interpolate(toks, size=(grid, grid), mode="bicubic", align_corners=False)
        return torch.cat([head, toks.reshape(1, d, grid * grid).transpose(1, 2)], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  # (B, N, D)
        b = x.shape[0]
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)  # [cls, patches]
        x = x + self._interpolate_pos_embed(x)
        if self.num_register_tokens:
            # 寄存器插在 cls 与 patches 之间（官方 prepare_tokens_with_masks 顺序）
            x = torch.cat([x[:, :1], self.register_tokens.expand(b, -1, -1), x[:, 1:]], dim=1)
        x = self.blocks(x)
        return self.norm(x)


def build_dinov2_vitl14(img_size: int = 518, num_register_tokens: int = 4) -> DinoV2:
    """官方 DINOv2 ViT-L/14 工厂（默认 4 寄存器，对应 dinov2_vitl14_reg 权重）。"""
    return DinoV2(
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        init_values=1.0,
    )


def infer_num_register_tokens(state: dict) -> int:
    """从 checkpoint 判定寄存器 token 数（``register_tokens`` 键缺失 → 0）。"""
    if not state or "register_tokens" not in state:
        return 0
    return int(state["register_tokens"].shape[1])
