# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""coreset 采样：torch 自写 SparseRandomProjection（JL 降维）+ 贪心 k-center。

对齐 anomalib/sklearn 的 ``SparseRandomProjection(eps=0.9)`` 语义，但不依赖 sklearn
（项目断网/去依赖）。投影仅用于选点，memory bank 存的是原始高维特征。
"""

from __future__ import annotations

import math

import torch


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.9) -> int:
    """eps-JL 嵌入所需的最小目标维度（sklearn 同款公式）。"""
    denom = (eps**2 / 2) - (eps**3 / 3)
    return int((4 * math.log(n_samples)) / denom)


class SparseRandomProjection:
    """稀疏随机投影：X (n, d) → X @ R.T (n, nc)，近似保持成对距离。

    Args:
        eps: JL 失真参数，默认 0.9（对齐 anomalib 参考）。
        n_components: 目标维度；None 时按 JL 下界自动计算。
        density: 稀疏度；None 时 1/sqrt(n_features)（对齐 sklearn/anomalib）。

    Note:
        全用全局 RNG（``torch.distributions.Binomial`` 不接受 ``generator=``），
        可复现性由调用方 ``torch.manual_seed`` 保证（train.py 的 set_seed 已满足）。
    """

    def __init__(self, eps: float = 0.9, n_components: int | None = None, density: float | None = None) -> None:
        self.eps = eps
        self.n_components = n_components
        self.density = density
        self.components_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "SparseRandomProjection":
        n_features = x.shape[1]
        nc = self.n_components or max(1, johnson_lindenstrauss_min_dim(x.shape[0], self.eps))
        nc = min(nc, n_features)
        density = self.density or (1.0 / math.sqrt(n_features))
        self.components_ = self._make_sparse_random_matrix(n_features, nc, density, x.device)
        self.n_components = nc
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.components_.T

    @staticmethod
    def _make_sparse_random_matrix(
        n_features: int, n_components: int, density: float, device: torch.device
    ) -> torch.Tensor:
        """构造 (n_components, n_features) 稀疏 ± 矩阵，缩放与 sklearn 一致。"""
        r = torch.zeros((n_components, n_features), dtype=torch.float32, device=device)
        nnz = torch.distributions.Binomial(total_count=n_features, probs=density).sample((n_components,))
        for i in range(n_components):
            k = int(nnz[i].item())
            if k == 0:
                continue
            cols = torch.randperm(n_features, device=device)[:k]  # 替代 sklearn sample_without_replacement
            vals = (torch.rand(k, device=device) < 0.5).float() * 2 - 1  # ±1
            r[i, cols] = vals
        return r.mul_(math.sqrt(1.0 / density) / math.sqrt(n_components))


class CoresetSampler:
    """k-center-greedy coreset 采样：返回使最大最近邻距离最小化的子集索引。"""

    def __init__(self, sampling_ratio: float = 0.1, eps: float = 0.9) -> None:
        if not 0.0 < sampling_ratio <= 1.0:
            raise ValueError(f"sampling_ratio 必须在 (0, 1]，收到 {sampling_ratio}")
        self.sampling_ratio = sampling_ratio
        self.projection = SparseRandomProjection(eps=eps)

    def sample_indices(self, features: torch.Tensor) -> torch.Tensor:
        """对 (N, C) 特征返回 coreset 索引 (k,)。bank 应取 features[idx] 原始特征。"""
        n = features.shape[0]
        k = max(1, int(n * self.sampling_ratio))
        if k >= n:
            return torch.arange(n, device=features.device)

        proj = self.projection.fit(features).transform(features)  # (N, nc≈300)

        indices = torch.zeros(k, dtype=torch.long, device=features.device)
        min_dist = torch.full((n,), float("inf"), device=features.device)
        current = int(torch.randint(0, n, (1,)).item())  # 随机起点（计入选集，对齐 anomalib gh-3459）
        for i in range(k):
            indices[i] = current
            d = torch.linalg.norm(proj - proj[current], ord=2, dim=1)
            min_dist = torch.minimum(min_dist, d)
            min_dist[current] = 0.0  # 排除已选点
            if i < k - 1:
                current = int(torch.argmax(min_dist).item())
        return indices
