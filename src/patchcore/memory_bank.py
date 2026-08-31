# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Memory bank：coreset 特征存储、归一化 scale、分块最近邻查询。

支持两种打分模式：
  - 标准 PatchCore：单全局 bank，query 逐 patch 对全 bank 做 cdist 最近邻（CUDA 半精度 GEMM 加速）。
  - PA-PatchCore（position-aware）：按空间位置建 bank，query 每个 patch 只在其邻域位置的
    bank 里找最近邻（缓解目标轻微偏移/错位），距离用逐元素 L2（比 cdist 展开式更稳）。
"""

from __future__ import annotations

from collections import defaultdict

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
        bank_dict: 可序列化 bank dict（bank / norm_scale / 可选 position_aware 等）。
        device: 查询设备；None 表示留在原始设备。
        inference_dtype: 半精度开关（None → CUDA 默认 FP16 查询；"float32" 禁用）。
    """

    def __init__(self, bank_dict: dict, device: torch.device | None = None, inference_dtype: str | None = None) -> None:
        self.device = device
        self.inference_dtype = inference_dtype
        self.bank = torch.from_numpy(np.asarray(bank_dict["bank"], dtype=np.float32))
        self.norm_scale = float(bank_dict.get("norm_scale") or 1.0)
        self.position_aware = bool(bank_dict.get("position_aware", False))
        self._bank_half = None
        self._cand_idx = None
        self._seg = None
        if self.device is not None:
            self.bank = self.bank.to(self.device)
            if not self.position_aware and resolve_inference_dtype(self.device.type, self.inference_dtype) != "float32":
                # 半精度 bank：kNN GEMM 走 tensor core；距离仍以 FP32 输出，打分路径不变
                # （PA 打分走 FP32 逐元素 L2，无需半精度 bank，省一半显存）
                self._bank_half = self.bank.half()
        if self.position_aware:
            self._build_neighborhood(bank_dict)

    def _build_neighborhood(self, bank_dict: dict) -> None:
        """预计算邻域候选：每个 query 位置 → 其邻域位置内所有 bank 行索引。"""
        h, w = bank_dict["grid"]
        nb = int(bank_dict.get("nb_size", 3))
        r = nb // 2
        n_patches = h * w
        bank_pos = torch.from_numpy(np.asarray(bank_dict["bank_pos"], dtype=np.int64))
        if self.device is not None:
            bank_pos = bank_pos.to(self.device)
        self.bank_pos = bank_pos
        # 每个位置 → 该位置的 bank 行索引（对称：3×3 邻域 q∈N(p) ⟺ p∈N(q)）
        rows_by_pos: dict[int, list[int]] = defaultdict(list)
        for k, pk in enumerate(bank_pos.tolist()):
            rows_by_pos[pk].append(k)
        cand_idx: list[int] = []
        seg: list[int] = []
        for q in range(n_patches):
            i, j = q // w, q % w
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        for row in rows_by_pos.get(ni * w + nj, ()):
                            cand_idx.append(row)
                            seg.append(q)
        self._cand_idx = torch.tensor(cand_idx, dtype=torch.long, device=self.device)
        self._seg = torch.tensor(seg, dtype=torch.long, device=self.device)

    def nearest_dist(self, query: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """query (nq, C) → (nq,) 最近邻距离；分块 cdist 控制峰值显存。

        CUDA 半精度时 query/bank 以 FP16 参与 cdist（``use_mm_for_euclid_dist`` 的 GEMM
        走 tensor core），结果 ``.float()`` 回 FP32，后续 score/exp/高斯平滑不受影响。
        """
        if self.device is not None:
            query = query.to(self.device)
        if self.position_aware:
            return self._nearest_dist_pa(query)
        bank = self._bank_half if self._bank_half is not None else self.bank
        q = query.half() if self._bank_half is not None else query
        out = []
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i : i + chunk], bank, compute_mode="use_mm_for_euclid_dist").float()
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    def _nearest_dist_pa(self, query: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """position-aware 邻域最近邻：逐元素 L2 + 分段 min（FP32，量级小无需半精度）。

        候选总数可达 9×K（每 query 位置 → 其邻域内所有 bank 行），一次性物化
        ``(候选数, C)`` 展开张量会 OOM，故按候选数分块、块内 ``index_reduce_`` 累积。
        """
        out = torch.full((query.shape[0],), float("inf"), dtype=torch.float32, device=self.device)
        seg = self._seg
        cand = self._cand_idx
        for i in range(0, seg.shape[0], chunk):
            s = seg[i : i + chunk]
            c = cand[i : i + chunk]
            q_e = query[s]
            b_e = self.bank[c]
            d = (q_e - b_e).square().sum(dim=1).sqrt()
            out.index_reduce_(0, s, d, reduce="amin", include_self=True)
        return out
