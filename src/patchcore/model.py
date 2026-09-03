# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""PatchCore 主类：训练建 bank / 推理打分 / 可选阈值校准 / checkpoint IO。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .anomaly_map import AnomalyMapGenerator
from .backbone import PatchBackbone
from .coreset import CoresetSampler
from .features import PatchFeatureExtractor
from .memory_bank import MemoryBank, compute_norm_scale
from .preprocess import PatchPreprocess
from .threshold import F1AdaptiveThreshold


@dataclass
class Prediction:
    """单图推理结果。

    image_score: 图像级分数 ∈ [0,1)，取热图 top-k 均值（k 见 PatchCore.score_topk_ratio）。
    anomaly_map: 像素级热图 (H,W) float32 ∈ [0,1)，与原图同尺寸。
    pred_label / pred_mask: 调用 ``fit_threshold`` 后才有；否则为 None。
    """

    image_score: float
    anomaly_map: np.ndarray
    pred_label: int | None = None
    pred_mask: np.ndarray | None = None


class PatchCore:
    """训练 / 推理 / 校准一体的 PatchCore 模型（torch 原生，无 anomalib 依赖）。

    Args:
        device: 运行设备（cuda 或 cpu）。
        backbone: ``torchvision.models`` 中任意主干名，如 dinov2_vitl14 /
            wide_resnet50_2（默认 dinov2_vitl14）。
        layers: 特征层名序列，如 ("blocks.6", "blocks.12", "blocks.18") 或 ("layer2", "layer3")。
        coreset_ratio: coreset 采样比例。bank 大小 = ratio × patch 总数，
            直接决定推理最近邻查询速度（调低即加速）。默认 0.1（参考口径）。
        score_topk_ratio: 图像级分数 top-k 均值比例（0 退化为 max）。默认 0.01。
        max_embed: 可选 patch 子采样安全阀；None（默认）= 全量采样。
        input_size / crop_size: 预处理缩放 / 中心裁剪，默认 518 / 518。
        pretrained_path: 离线预训练权重路径（train 用）；None 时 predict 从 shared.pth 加载。
        sigma: 热图高斯平滑标准差。
    """

    def __init__(
        self,
        device: torch.device,
        backbone: str = "dinov2_vitl14",
        layers=("blocks.6", "blocks.12", "blocks.18"),
        coreset_ratio: float = 0.1,
        score_topk_ratio: float = 0.01,
        max_embed: int | None = None,
        input_size=(518, 518),
        crop_size=(518, 518),
        pretrained_path: Path | str | None = None,
        sigma: float = 4.0,
    ) -> None:
        self.device = device
        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.coreset_ratio = coreset_ratio
        self.score_topk_ratio = score_topk_ratio
        self.max_embed = max_embed
        self.sigma = sigma
        self.backbone = PatchBackbone(device, pretrained_path=pretrained_path, name=backbone, layers=self.layers)
        self.extractor = PatchFeatureExtractor(self.backbone, self.layers)
        self.preprocess = PatchPreprocess(input_size, crop_size)
        self.bank_dict: dict | None = None
        self.image_threshold: float | None = None
        self._bank: MemoryBank | None = None
        self.onnx = None  # 可选 OnnxBackbone；非 None 时 predict 走 ONNX 特征提取

    # ------------------------------------------------------------------ 训练
    def fit(self, image_paths: list[Path]) -> dict:
        """从正常样本构建类别 memory bank，返回可序列化 bank_dict。

        复用同一 backbone 实例，可跨类别循环调用（每类只重建 bank）。
        """
        feats = []
        for p in image_paths:
            x = self.preprocess.encode(Image.open(p)).to(self.device)
            patch = self.extractor(x)
            # 高分辨率下每图 patch 多，逐图特征先挪 CPU 累积，显存只留 coreset 子集
            feats.append(patch.reshape(patch.shape[0], -1).T.detach().cpu())
        all_feats = torch.cat(feats, dim=0)
        del feats

        if self.max_embed is not None and all_feats.shape[0] > self.max_embed:
            perm = torch.randperm(all_feats.shape[0])[: self.max_embed]
            all_feats = all_feats[perm]

        all_feats = all_feats.to(self.device)

        if self.coreset_ratio < 1.0:
            idx = CoresetSampler(self.coreset_ratio).sample_indices(all_feats)
            bank = all_feats[idx]
            coreset_indices = idx.cpu().tolist()
        else:
            bank = all_feats
            coreset_indices = None

        norm_scale = compute_norm_scale(all_feats, bank)
        self.bank_dict = {
            "bank": bank.cpu().numpy().astype(np.float32),
            "coreset_indices": coreset_indices,
            "norm_scale": norm_scale,
            "feature_dim": int(bank.shape[1]),
            "backbone": self.backbone_name,
            "layers": list(self.layers),
            "input_size": list(self.preprocess.input_size),
            "crop_size": list(self.preprocess.crop_size),
            "sigma": self.sigma,
        }
        # 全量 patch 特征只用于建 bank，之后立刻释放，控制跨类别训练显存峰值
        del all_feats, bank
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._bank = None
        return self.bank_dict

    # ------------------------------------------------------------------ 推理
    def predict_map(self, image: Image.Image) -> np.ndarray:
        """单图推理，返回像素级异常图 (H, W) float32 ∈ [0,1)，不含图像级分数。

        与 ``predict`` 拆开：多视角 TTA 可对每视角热图翻转对齐后平均，
        再用 ``score_from_map`` 统一汇总图像级分数。预处理/平滑参数从 bank_dict 重建。
        """
        if self.bank_dict is None:
            raise RuntimeError("未设置 bank（先调用 fit 或 load_category）")
        if self._bank is None:
            self._bank = MemoryBank(self.bank_dict, device=self.device)
        preprocess = PatchPreprocess.from_dict(self.bank_dict)
        map_gen = AnomalyMapGenerator(sigma=self.bank_dict.get("sigma") or 4.0)

        if self.onnx is not None:
            # ONNX 路径：主干特征由 ONNX Runtime 提取，聚合逻辑与 PyTorch 一致
            x_np = preprocess.encode(image).numpy()[None].astype("float32")  # (1,3,H,W)
            named = {
                name: torch.from_numpy(f).to(self.device)
                for name, f in zip(self.layers, self.onnx(x_np))
            }
            patch = self.extractor.aggregate(named)  # (C, h, w)
        else:
            x = preprocess.encode(image).to(self.device)
            patch = self.extractor(x)  # (C, h, w)
        h, w = patch.shape[-2:]
        q = patch.reshape(patch.shape[0], -1).T  # (h*w, C)
        dist = self._bank.nearest_dist(q)  # (h*w,)
        score_map = dist.reshape(1, h, w)
        score_map = map_gen(score_map, image.size[::-1])[0]  # (H, W)
        score_map = 1.0 - torch.exp(-score_map / self._bank.norm_scale)
        return score_map.cpu().numpy().astype(np.float32)

    def score_from_map(self, anomaly_map: np.ndarray) -> float:
        """图像级分数 = 异常图 top-k 均值（k = score_topk_ratio × 像素数，至少 1）。

        用 top-k 均值替代单一 max：避免正常图里孤立噪声 patch 把分数顶高，
        又比整图均值更能保留细微缺陷（裂痕/脏污）的局部峰值。
        ``score_topk_ratio <= 0`` 退化为 max（A/B 对照用）。
        """
        flat = np.asarray(anomaly_map, dtype=np.float32).ravel()
        if self.score_topk_ratio <= 0:
            return float(flat.max())
        k = max(1, int(flat.size * self.score_topk_ratio))
        return float(np.partition(flat, -k)[-k:].mean())

    def predict(self, image: Image.Image) -> Prediction:
        """单图推理。返回图像级分数（top-k 均值）+ 像素级异常图。"""
        anomaly_map = self.predict_map(image)
        image_score = self.score_from_map(anomaly_map)
        pred = Prediction(image_score=image_score, anomaly_map=anomaly_map)
        if self.image_threshold is not None:
            pred.pred_label = int(image_score >= self.image_threshold)
            pred.pred_mask = F1AdaptiveThreshold.apply(anomaly_map, self.image_threshold)
        return pred

    # ------------------------------------------------------------------ 阈值
    def fit_threshold(self, image_paths: list[Path], labels, num_bins: int = 200) -> float:
        """在带标签图上拟合 F1-adaptive 图像级阈值（开发期验证用，含异常样本才有意义）。"""
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
        # bank_dict 同样携带架构信息，双保险（与 shared.pth 一致时无操作）
        backbone = self.bank_dict.get("backbone") or "dinov2_vitl14"
        layers = tuple(self.bank_dict.get("layers") or ("blocks.6", "blocks.12", "blocks.18"))
        if backbone != self.backbone_name or layers != self.layers:
            self.backbone = PatchBackbone(self.device, name=backbone, layers=layers)
            self.backbone_name = backbone
            self.layers = layers
            self.extractor = PatchFeatureExtractor(self.backbone, self.layers)
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
        # 架构信息随 ckpt 走，predict 端无需知道训练时的 backbone 名
        backbone = ckpt.get("backbone") or "dinov2_vitl14"
        layers = tuple(ckpt.get("layers") or ("blocks.6", "blocks.12", "blocks.18"))
        if backbone != self.backbone_name or layers != self.layers:
            self.backbone = PatchBackbone(self.device, name=backbone, layers=layers)
            self.backbone_name = backbone
            self.layers = layers
            self.extractor = PatchFeatureExtractor(self.backbone, self.layers)
        self.backbone.load_state(ckpt["state_dict"])

    # ------------------------------------------------------------- ONNX 加速
    def export_onnx(self, path: Path, input_size=(224, 224)) -> None:
        """导出截断主干为 ONNX（输出为逐层特征，顺序 = self.layers）。"""
        self.backbone.export_onnx(path, input_size=input_size)

    def load_onnx(self, path: Path) -> None:
        """加载 ONNX 主干；之后 predict 走 ONNX 特征提取（聚合逻辑不变）。"""
        from .onnx import OnnxBackbone

        self.onnx = OnnxBackbone(path)
