# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""级联剪枝多级检索：浅层全量 → 中层子集 → 深层极小集。

对 query patch：浅层 bank 全量 cdist 取 top-k1 最近邻（bank 行号），中层只在
存活行号 gather 的子集里取 top-k2，深层在小候选集取 min 距离作为异常分数。
各层 bank 行索引由 coreset 在 concat 特征上选一次索引并应用到每层，跨层对齐，
浅层选中的行号可直接映射到中层/深层 bank。

matryoshka 降算：``prefix_dims`` 可选地对浅/中层做前缀维切片（默认关闭=全维精确排序）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .memory_bank import MemoryBank

# 峰值显存预算（字节），用于动态分块
_PEAK_BYTES = 256 << 20


class CascadeMemoryBank:
    """多层 bank + 级联 top-k 剪枝查询。

    Args:
        bank_dict: format_version=2 新 bank dict（含 ``banks``/``levels``/``cascade``）。
        device: 查询设备；None 表示留在原始设备。
    """

    def __init__(self, bank_dict: dict, device: torch.device | None = None) -> None:
        self.device = device
        self.levels = list(bank_dict["levels"])
        self.banks: dict[str, Tensor] = {}
        for lv in self.levels:
            t = torch.from_numpy(np.asarray(bank_dict["banks"][lv], dtype=np.float32))
            self.banks[lv] = t.to(device) if device is not None else t
        self.norm_scale = float(bank_dict.get("norm_scale") or 1.0)
        cascade = bank_dict.get("cascade") or {}
        self.k1_ratio = float(cascade.get("k1_ratio", 0.1))
        self.k2_ratio = float(cascade.get("k2_ratio", 0.1))
        self.prefix_dims = bank_dict.get("prefix_dims") or {}
        m = self.banks[self.levels[0]].shape[0]
        self.k1 = min(m, max(1, int(math.ceil(self.k1_ratio * m))))
        self.k2 = min(self.k1, max(1, int(math.ceil(self.k2_ratio * self.k1))))

    def _dist(self, q: Tensor, bank: Tensor, dims: int | None) -> Tensor:
        """欧氏距离（可选前缀维切片），mm 路径省内存。"""
        if dims is not None and dims < bank.shape[1]:
            q, bank = q[:, :dims], bank[:, :dims]
        return torch.cdist(q, bank, compute_mode="use_mm_for_euclid_dist")

    def _chunk(self, k: int, dim: int) -> int:
        """按 level1 gather 峰值 (c×k×dim×4B) 估算查询块大小。"""
        return max(1, min(512, int(_PEAK_BYTES // (k * dim * 4))))

    def __call__(self, query: dict[str, Tensor]) -> Tensor:
        """query {level: (nq, C)} → (nq,) 级联最近邻距离（深层 min）。"""
        if len(self.levels) < 2:
            raise ValueError("CascadeMemoryBank 至少需要 2 个特征层")
        l0, l1, l2 = self.levels[0], self.levels[1], self.levels[-1]
        b0, b1, b2 = self.banks[l0], self.banks[l1], self.banks[l2]
        q0 = query[l0].to(self.device) if self.device is not None else query[l0]
        q1 = query[l1].to(self.device) if self.device is not None else query[l1]
        q2 = query[l2].to(self.device) if self.device is not None else query[l2]
        d0_dims = self.prefix_dims.get(l0)
        d1_dims = self.prefix_dims.get(l1)
        chunk = self._chunk(self.k1, b1.shape[1])
        nq = q0.shape[0]
        out: list[Tensor] = []
        for start in range(0, nq, chunk):
            qc0 = q0[start : start + chunk]
            c = qc0.shape[0]
            # 浅层全量 → top-k1 bank 行号
            d0 = self._dist(qc0, b0, d0_dims)
            _, idx0 = d0.topk(self.k1, dim=1, largest=False)  # (c, k1) bank 行号
            # 中层子集 → top-k2（映射回全局行号）
            b1_sub = b1[idx0]  # (c, k1, D1) gather
            qc1 = q1[start : start + c]
            d1 = self._dist(qc1.unsqueeze(1), b1_sub, d1_dims)[:, 0, :]
            _, idx1 = d1.topk(self.k2, dim=1, largest=False)  # (c, k2)
            idxg = idx0.gather(1, idx1)  # (c, k2) 全局行号
            # 深层极小集 → min 距离
            qc2 = q2[start : start + c]
            d2 = self._dist(qc2.unsqueeze(1), b2[idxg], None)[:, 0, :]
            out.append(d2.min(dim=1).values)
        return torch.cat(out)


def build_bank(bank_dict: dict, device: torch.device | None = None):
    """按 bank_dict 格式分派：新级联（``banks``）走 CascadeMemoryBank，旧单 bank 走 MemoryBank。"""
    if "banks" in bank_dict:
        return CascadeMemoryBank(bank_dict, device=device)
    return MemoryBank(bank_dict, device=device)


def make_bank_dict(
    banks: dict[str, np.ndarray],
    levels,
    cascade_ratios,
    use_prefix_dist: bool,
    prefix_dims: dict | None,
    coreset_indices,
    norm_scale: float,
    backbone: str,
    layers,
    input_size,
    crop_size,
    sigma: float,
) -> dict:
    """组装 format_version=2 bank dict（fit 用；predict 端重建级联查询）。"""
    return {
        "format_version": 2,
        "banks": banks,
        "levels": list(levels),
        "cascade": {"k1_ratio": float(cascade_ratios[0]), "k2_ratio": float(cascade_ratios[1])},
        "use_prefix_dist": bool(use_prefix_dist),
        "prefix_dims": prefix_dims,
        "coreset_indices": coreset_indices,
        "coreset_concat_dim": int(banks[levels[0]].shape[1] * len(levels)),
        "norm_scale": float(norm_scale),
        "feature_dim": int(banks[levels[0]].shape[1]),
        "backbone": backbone,
        "layers": list(layers),
        "input_size": list(input_size),
        "crop_size": list(crop_size),
        "sigma": float(sigma),
    }
