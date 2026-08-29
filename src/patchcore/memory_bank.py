# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Memory bank：coreset 特征存储、归一化 scale、分块最近邻查询。"""

from __future__ import annotations

import numpy as np
import torch

from .features import resolve_inference_dtype


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
        inference_dtype: 半精度开关（None → CUDA 默认 FP16 查询；"float32" 禁用）。
    """

    def __init__(self, bank_dict: dict, device: torch.device | None = None, inference_dtype: str | None = None) -> None:
        self.device = device
        self.inference_dtype = inference_dtype
        self.bank = torch.from_numpy(np.asarray(bank_dict["bank"], dtype=np.float32))
        self.norm_scale = float(bank_dict.get("norm_scale") or 1.0)
        self._bank_half = None
        if self.device is not None:
            self.bank = self.bank.to(self.device)
            if resolve_inference_dtype(self.device.type, self.inference_dtype) != "float32":
                # 半精度 bank：kNN GEMM 走 tensor core；距离仍以 FP32 输出，打分路径不变
                self._bank_half = self.bank.half()

    def nearest_dist(self, query: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """query (nq, C) → (nq,) 最近邻距离；分块 cdist 控制峰值显存。

        CUDA 半精度时 query/bank 以 FP16 参与 cdist（``use_mm_for_euclid_dist`` 的 GEMM
        走 tensor core），结果 ``.float()`` 回 FP32，后续 score/exp/高斯平滑不受影响。
        """
        if self.device is not None:
            query = query.to(self.device)
        bank = self._bank_half if self._bank_half is not None else self.bank
        q = query.half() if self._bank_half is not None else query
        out = []
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i : i + chunk], bank, compute_mode="use_mm_for_euclid_dist").float()
            out.append(d.min(dim=1).values)
        return torch.cat(out)
