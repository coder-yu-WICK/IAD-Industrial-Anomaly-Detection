# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PA-PatchCore：位置感知 PatchCore（position-wise 自适应 coreset bank + 邻域打分）。

对齐论文要点（PA-PatchCore, 精密工学会 JSPE 2025）：
  1. 按空间位置分组建 bank（而非单一全局 bank）；
  2. 用 variation map 按各位置正常特征的变化程度自适应分配采样率（变化大 → 多采）；
  3. 打分时同时参考邻域位置的 bank，缓解检测目标轻微偏移/错位。

仅在 PatchCore 基础上覆盖 fit()（bank 构建）；推理端 MemoryBank 依 bank_dict 的
``position_aware`` 标记自动走邻域打分，主干 / 预处理 / 阈值 / IO 全部复用 PatchCore。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .coreset import CoresetSampler
from .model import PatchCore


class PatchCorePA(PatchCore):
    """PA-PatchCore：position-wise bank + variation-map 自适应采样 + 邻域打分。"""

    def fit(self, image_paths: list[Path], nb_size: int = 3) -> dict:
        """从正常样本构建 position-wise 自适应 memory bank。

        Args:
            image_paths: 正常样本路径。
            nb_size: 邻域边长（奇数，默认 3 表示 3×3 邻域，含自身）。
        """
        # 逐图 patch 特征 (h*w, C)，按位置堆叠成 (N, h*w, C)，全程留在 CPU 控制显存
        feats = []
        h = w = None
        for p in image_paths:
            x = self.preprocess.encode(Image.open(p)).to(self.device)
            patch = self.extractor(x)  # (C, h, w)
            if h is None:
                _, h, w = patch.shape
            feats.append(patch.reshape(patch.shape[0], -1).T.detach().cpu())  # (h*w, C)
        all_feats = torch.stack(feats, dim=0)  # (N, n_patches, C) on CPU
        del feats
        n_img, n_patches, c = all_feats.shape

        # 1) variation map：每个位置正常特征跨图方差（变化大 → 需要更多代表点）
        var = all_feats.var(dim=0)          # (n_patches, C)
        variation = var.mean(dim=1)          # (n_patches,)
        base = self.coreset_ratio
        w_map = variation / (variation.mean() + 1e-8)  # 均值 ~1
        r_map = (base * w_map).clamp(min=base * 0.2, max=min(1.0, base * 5.0))
        k_map = (r_map * n_img).clamp(min=1, max=n_img).round().long()  # (n_patches,)

        # 2) 逐位置 coreset：采样率随 variation 自适应（比全局 coreset 更省显存）
        bank_parts: list[torch.Tensor] = []
        bank_pos: list[torch.Tensor] = []
        for p in range(n_patches):
            pos_feats = all_feats[:, p, :].to(self.device)  # (N, C)
            k = int(k_map[p].item())
            if k >= n_img:
                idx = torch.arange(n_img, device=self.device)
            else:
                idx = CoresetSampler(k / n_img).sample_indices(pos_feats)
            bank_parts.append(pos_feats[idx])
            bank_pos.append(torch.full((idx.numel(),), p, dtype=torch.long, device=self.device))
        bank = torch.cat(bank_parts, dim=0)    # (K, C)
        bank_pos = torch.cat(bank_pos, dim=0)  # (K,)

        # 3) 归一化 scale：全量训练 patch 到 bank 最近邻距离均值（分块搬到 GPU，控显存）
        norm_scale = self._pa_norm_scale(all_feats.reshape(-1, c), bank)

        self.bank_dict = {
            "bank": bank.cpu().numpy().astype(np.float32),
            "bank_pos": bank_pos.cpu().numpy().astype(np.int64),
            "position_aware": True,
            "grid": [h, w],
            "nb_size": nb_size,
            "norm_scale": norm_scale,
            "feature_dim": int(c),
            "backbone": self.backbone_name,
            "layers": list(self.layers),
            "input_size": list(self.preprocess.input_size),
            "crop_size": list(self.preprocess.crop_size),
            "sigma": self.sigma,
        }
        self._bank = None
        return self.bank_dict

    @staticmethod
    def _pa_norm_scale(feats_cpu: torch.Tensor, bank: torch.Tensor, chunk: int = 512) -> float:
        """分块把训练特征搬到 GPU 算最近邻，避免整张 (N*n_patches, C) 常驻显存。"""
        dists = []
        for i in range(0, feats_cpu.shape[0], chunk):
            d = torch.cdist(
                feats_cpu[i : i + chunk].to(bank.device), bank, compute_mode="use_mm_for_euclid_dist"
            )
            dists.append(d.min(dim=1).values.cpu())
        scale = float(torch.cat(dists).mean())
        return max(scale, 1e-6)
