# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主类：级联多级检索建 bank / 推理打分 / 可选阈值校准 / ckpt IO。

架构（format_version=2）：逐层 bank（浅/中/深），级联 top-k 剪枝查询，跨层索引对齐。
兼容 format_version=1（旧单 concat bank）ckpt：predict 走 concat_feature + MemoryBank。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .anomaly_map import AnomalyMapGenerator
from .backbone import PatchBackbone
from .cascade import CascadeMemoryBank, build_bank, make_bank_dict
from .coreset import CoresetSampler
from .features import PatchFeatureExtractor
from .memory_bank import compute_norm_scale
from .preprocess import PatchPreprocess
from .threshold import F1AdaptiveThreshold


@dataclass
class Prediction:
    """单图推理结果：image_score ∈[0,1) 取热图最大值；anomaly_map (H,W) float32。"""

    image_score: float
    anomaly_map: np.ndarray
    pred_label: int | None = None
    pred_mask: np.ndarray | None = None


class PatchCore:
    """训练 / 推理 / 校准一体的 PatchCore（torch 原生，无 anomalib 依赖）。

    Args:
        device: 运行设备。
        backbone: 主干名；默认 ``franca_vitb14``（包内 DINOv2 风格 ViT-B/14）。
            torchvision 原生名（如 vit_b_16）仅用于加载旧 ckpt 的兼容。
        layers: 级联特征层名，浅→深顺序（默认 blocks.3/6/9）。
        coreset_ratio: coreset 采样比例（每层 bank = ratio × patch 总数）。
        max_embed: 每类参与 coreset 的 patch 子采样上限（518 网格 n≈315k，必设）。
        input_size / crop_size: 预处理缩放 / 中心裁剪，默认 518（Franca 预训练分辨率）。
        pretrained_path: 离线预训练权重路径（train 用）；None 时 predict 从 shared.pth 加载。
        sigma: 热图高斯平滑标准差。
        cascade_ratios: 级联保留比例 (k1_ratio, k2_ratio)，默认 0.1/0.1。
        use_prefix_dist / prefix_dims: matryoshka 前缀维降算（默认关闭=全维精确排序）。
    """

    def __init__(
        self,
        device: torch.device,
        backbone: str = "franca_vitb14",
        layers=("blocks.3", "blocks.6", "blocks.9"),
        coreset_ratio: float = 0.1,
        max_embed: int | None = 50000,
        input_size=(518, 518),
        crop_size=(518, 518),
        pretrained_path: Path | str | None = None,
        sigma: float = 4.0,
        cascade_ratios=(0.1, 0.1),
        use_prefix_dist: bool = False,
        prefix_dims: dict | None = None,
        cascade: bool = True,
    ) -> None:
        self.device = device
        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.coreset_ratio = coreset_ratio
        self.max_embed = max_embed
        self.sigma = sigma
        self.cascade_ratios = tuple(cascade_ratios)
        self.use_prefix_dist = use_prefix_dist
        self.prefix_dims = dict(prefix_dims) if prefix_dims else None
        # True=多层级联 bank（层维嵌套）；False=单 concat bank（无层维嵌套，
        # 消融基线用；配合 prefix_dims 可做"仅维度嵌套"）。
        self.cascade = cascade
        self.backbone = PatchBackbone(device, pretrained_path=pretrained_path, name=backbone, layers=self.layers)
        self.extractor = PatchFeatureExtractor(self.backbone, self.layers)
        self.preprocess = PatchPreprocess(input_size, crop_size)
        self.bank_dict: dict | None = None
        self.image_threshold: float | None = None
        self._bank = None

    # ------------------------------------------------------------------ 训练
    def fit(self, image_paths: list[Path]) -> dict:
        """从正常样本构建逐层级联 bank，返回可序列化 bank_dict。

        逐层特征行序 = 同一图像序 × 同一网格 row-major token 序，跨层对齐；
        coreset 在 concat 特征上选一次索引，同索引应用到每层 bank。
        """
        per_level: dict[str, list] = {}
        for p in image_paths:
            x = self.preprocess.encode(Image.open(p)).to(self.device)
            for lv, f in self.extractor(x).items():  # {layer: (C, h, w)}
                per_level.setdefault(lv, []).append(f.reshape(f.shape[0], -1).T)
        all_feats = {lv: torch.cat(v, dim=0) for lv, v in per_level.items()}

        if self.max_embed is not None and all_feats[self.layers[0]].shape[0] > self.max_embed:
            perm = torch.randperm(all_feats[self.layers[0]].shape[0])[: self.max_embed]
            all_feats = {lv: f[perm] for lv, f in all_feats.items()}

        if not self.cascade:
            # ---- 单 concat bank（legacy 格式，无层维嵌套）----
            # 可选前缀维切片（matryoshka 维度嵌套）：prefix_dims 的 "concat" 键
            # 指定拼接特征保留的前缀维数，用于"仅维度嵌套"消融。
            eff = torch.cat([all_feats[lv] for lv in self.layers], dim=1)
            if self.use_prefix_dist:
                pd = self._single_prefix_dim(eff.shape[1])
                eff = eff[:, :pd]
            if self.coreset_ratio < 1.0:
                idx = CoresetSampler(self.coreset_ratio).sample_indices(eff)
            else:
                idx = torch.arange(eff.shape[0])
            bank = eff[idx]
            norm_scale = compute_norm_scale(eff, bank)
            self.bank_dict = {
                "format_version": 1,
                "bank": bank.cpu().numpy().astype(np.float32),
                "norm_scale": norm_scale,
                "prefix_dim": int(eff.shape[1]),
                "backbone": self.backbone_name,
                "layers": list(self.layers),
                "input_size": list(self.preprocess.input_size),
                "crop_size": list(self.preprocess.crop_size),
                "sigma": self.sigma,
            }
            self._bank = None
            return self.bank_dict

        concat = torch.cat([all_feats[lv] for lv in self.layers], dim=1)
        if self.coreset_ratio < 1.0:
            idx = CoresetSampler(self.coreset_ratio).sample_indices(concat)
            coreset_indices = idx.cpu().tolist()
        else:
            idx = torch.arange(concat.shape[0])
            coreset_indices = None

        deep = self.layers[-1]
        norm_scale = compute_norm_scale(all_feats[deep], all_feats[deep][idx])
        banks = {lv: all_feats[lv][idx].cpu().numpy().astype(np.float32) for lv in self.layers}
        self.bank_dict = make_bank_dict(
            banks, self.layers, self.cascade_ratios, self.use_prefix_dist, self.prefix_dims,
            coreset_indices, norm_scale, self.backbone_name, self.layers,
            self.preprocess.input_size, self.preprocess.crop_size, self.sigma,
        )
        self._bank = None
        return self.bank_dict

    def _single_prefix_dim(self, total_dim: int) -> int:
        """单 bank 模式的前缀维：取 prefix_dims['concat']，缺省则全维。"""
        pd = self.prefix_dims.get("concat") if self.prefix_dims else None
        if pd is None:
            return total_dim
        pd = int(pd)
        return max(1, min(pd, total_dim))

    # ------------------------------------------------------------------ 推理
    def predict(self, image: Image.Image) -> Prediction:
        """单图推理。预处理/平滑/级联参数一律从 bank_dict 重建，保证与训练一致。"""
        if self.bank_dict is None:
            raise RuntimeError("未设置 bank（先调用 fit 或 load_category）")
        if self._bank is None:
            self._bank = build_bank(self.bank_dict, device=self.device)
        preprocess = PatchPreprocess.from_dict(self.bank_dict)
        map_gen = AnomalyMapGenerator(sigma=self.bank_dict.get("sigma") or 4.0)

        x = preprocess.encode(image).to(self.device)
        if isinstance(self._bank, CascadeMemoryBank):
            feats = self.extractor(x)
            query = {lv: f.reshape(f.shape[0], -1).T for lv, f in feats.items()}
            h, w = feats[self.layers[0]].shape[-2:]
            dist = self._bank(query)  # (h*w,)
        else:  # legacy 单 concat bank
            patch = self.extractor.concat_feature(x)
            h, w = patch.shape[-2:]
            pd = (self.bank_dict or {}).get("prefix_dim")
            if pd is not None:
                patch = patch[:pd]
            dist = self._bank.nearest_dist(patch.reshape(patch.shape[0], -1).T)

        score_map = dist.reshape(1, h, w)
        score_map = map_gen(score_map, image.size[::-1])[0]  # (H, W)
        score_map = 1.0 - torch.exp(-score_map / self._bank.norm_scale)
        anomaly_map = score_map.cpu().numpy().astype(np.float32)
        pred = Prediction(image_score=float(anomaly_map.max()), anomaly_map=anomaly_map)
        if self.image_threshold is not None:
            pred.pred_label = int(pred.image_score >= self.image_threshold)
            pred.pred_mask = F1AdaptiveThreshold.apply(anomaly_map, self.image_threshold)
        return pred

    # ------------------------------------------------------------------ 阈值
    def fit_threshold(self, image_paths: list[Path], labels, num_bins: int = 200) -> float:
        """在带标签图上拟合 F1-adaptive 图像级阈值（开发期验证用）。"""
        scores = np.asarray([self.predict(Image.open(p)).image_score for p in image_paths], dtype=np.float32)
        self.image_threshold = F1AdaptiveThreshold().fit(scores, labels, num_bins=num_bins)
        return self.image_threshold

    # ------------------------------------------------------------- ckpt IO
    def save_category(self, path: Path, category: str, bank_dict: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"category": category, "bank_dict": bank_dict or self.bank_dict}, path)

    def load_category(self, path: Path) -> dict:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.bank_dict = data["bank_dict"]
        self._rebuild_if_needed()
        self._bank = None
        return self.bank_dict

    def save_shared(self, path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.backbone.state_dict(),
                "seed": seed,
                "backbone": self.backbone_name,
                "layers": list(self.layers),
            },
            path,
        )

    def load_shared(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._rebuild_if_needed(backbone=ckpt.get("backbone"), layers=ckpt.get("layers"))
        self.backbone.load_state(ckpt["state_dict"])

    # 按 ckpt 记录重建主干/提取器（predict 端无需知道训练时的 backbone 名）
    def _rebuild_if_needed(self, backbone=None, layers=None) -> None:
        src = self.bank_dict or {}
        backbone = backbone or src.get("backbone") or "franca_vitb14"
        layers = tuple(layers or src.get("layers") or ("blocks.3", "blocks.6", "blocks.9"))
        if backbone != self.backbone_name or layers != self.layers:
            self.backbone = PatchBackbone(self.device, name=backbone, layers=layers)
            self.backbone_name = backbone
            self.layers = layers
            self.extractor = PatchFeatureExtractor(self.backbone, self.layers)
