# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Memory bank：coreset 特征存储、归一化 scale、分块最近邻查询。"""

from __future__ import annotations

import numpy as np
import torch


def compute_norm_scale(feats: torch.Tensor, bank: torch.Tensor, chunk: int = 256) -> float:
    """归一化 scale = 全量训练 patch 到 bank 最近邻距离均值（分块算，防爆内存）。"""
    dists = []
    for i in range(0, feats.shape[0], chunk):
        d = torch.cdist(feats[i : i + chunk], bank, compute_mode="use_mm_for_euclid_dist")
        dists.append(d.min(dim=1).values)
    scale = float(torch.cat(dists).mean())
    return max(scale, 1e-6)


class MemoryBank:
    """存储 coreset 特征并提供分块最近邻查询。

    Args:
        bank_dict: 可序列化 bank dict（bank / norm_scale 等）。
        device: 查询设备；None 表示留在原始设备。
    """

    def __init__(self, bank_dict: dict, device: torch.device | None = None) -> None:
        self.device = device
        self.bank = torch.from_numpy(np.asarray(bank_dict["bank"], dtype=np.float32))
        self.norm_scale = float(bank_dict.get("norm_scale") or 1.0)
        if self.device is not None:
            self.bank = self.bank.to(self.device)

    def nearest_dist(self, query: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """query (nq, C) → (nq,) 最近邻距离；分块 cdist 控制峰值显存。"""
        if self.device is not None:
            query = query.to(self.device)
        out = []
        for i in range(0, query.shape[0], chunk):
            d = torch.cdist(query[i : i + chunk], self.bank, compute_mode="use_mm_for_euclid_dist")
            out.append(d.min(dim=1).values)
        return torch.cat(out)
