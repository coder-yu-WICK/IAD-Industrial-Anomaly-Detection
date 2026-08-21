# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Omni-AD Data Module"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, TestSplitMode, ValSplitMode

from . import sample_manifest

logger = logging.getLogger(__name__)

TRAIN_MANIFEST = "train_manifest.csv"
TEST_MANIFEST = "test_manifest.csv"


def make_omniad_samples(
    root: Path | str,
    manifest: str,
    split: str | Split,
    category: str | None = None,
    task: str | TaskType = TaskType.SEGMENTATION,
) -> pd.DataFrame:
    """从 manifest CSV 构建 samples DataFrame（image_path/split/label/label_index/mask_path + attrs["task"]）。"""
    root = Path(root)
    manifest_path = root / manifest
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if category is not None:
        df = df[df["category"] == category]
    if df.empty:
        raise RuntimeError(f"manifest 中无样本: category={category} manifest={manifest_path}")

    if "anomaly" in df.columns:
        anomalous = df["anomaly"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        df["label_index"] = anomalous.map({True: LabelName.ABNORMAL, False: LabelName.NORMAL})
        df["label"] = anomalous.map({True: "abnormal", False: "good"})
    else:
        df["label_index"] = LabelName.NORMAL
        df["label"] = "good"

    df["image_path"] = df["image_path"].map(lambda p: str(root / str(p).strip()))
    if "ground_truth" in df.columns:
        has_gt = df["ground_truth"].notna() & df["ground_truth"].astype(str).str.strip().ne("")
        df["mask_path"] = ""
        df.loc[has_gt, "mask_path"] = df.loc[has_gt, "ground_truth"].map(lambda p: str(root / str(p).strip()))
    else:
        df["mask_path"] = ""

    df["split"] = split.value if isinstance(split, Split) else split
    df.attrs["task"] = task.value if isinstance(task, TaskType) else str(task)
    return df.reset_index(drop=True)


class OMNIADDataset(AnomalibDataset):
    def __init__(
        self,
        root: Path | str,
        manifest: str,
        split: str | Split,
        category: str | None = None,
        task: str | TaskType = TaskType.SEGMENTATION,
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.root = Path(root)
        self.category = category
        self.samples = make_omniad_samples(self.root, manifest, split, category=category, task=task)


class OMNIAD(AnomalibDataModule):
    """Omni-AD Datamodule（manifest CSV 驱动，与 ``MVTecAD`` 同父类，可传给 ``Engine``/``Trainer``）。

    Args:
        root: 数据根目录（manifest 与各类别目录所在处）。默认 repo 的 ``data/Omni-AD-30-release``。
        category: 类别名（如 ``"air_conditioner_filter"``）；``None`` 加载全部类别。默认 ``None``。
        task: 任务类型，``segmentation``（默认）或 ``classification``。
        train_batch_size / eval_batch_size / num_workers: batch 与 worker 数。默认 32 / 32 / 8。
        train_augmentations / val_augmentations / test_augmentations / augmentations: 各阶段增强，
            未指定某阶段时用 augmentations。默认 ``None``。
        test_split_mode / test_split_ratio / val_split_mode / val_split_ratio / seed: 数据切分，
            默认与 ``MVTecAD`` 一致（FROM_DIR / 0.2 / SAME_AS_TEST / 0.5）。

        >>> datamodule = OMNIAD(root="./data/Omni-AD-30-release", category=None)  # None=全部类别
        >>> datamodule.setup()
        >>> len(datamodule.train_data), len(datamodule.test_data)
        (2252, 2212)
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: str | TaskType = TaskType.SEGMENTATION,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root or Path(__file__).resolve().parents[2] / "data" / "Omni-AD-30-release")
        self.category = category
        self.task_type = TaskType(task) if isinstance(task, str) else task  # 基类 task 为只读属性

    def _ensure_manifests(self) -> None:
        """manifest 缺失时直接调用 sample_manifest 生成。"""
        if all((self.root / m).exists() for m in (TRAIN_MANIFEST, TEST_MANIFEST)):
            return
        if not self.root.is_dir():
            raise NotADirectoryError(f"--data-root 不存在: {self.root}")
        logger.info("[OMNIAD] manifest 缺失，调用 sample_manifest.generate_manifests 生成 ...")
        sample_manifest.generate_manifests(self.root)

    def prepare_data(self) -> None:
        self._ensure_manifests()

    def _setup(self, _stage: str | None = None) -> None:
        self._ensure_manifests()
        self.train_data = OMNIADDataset(self.root, TRAIN_MANIFEST, Split.TRAIN, self.category, self.task_type)
        self.test_data = OMNIADDataset(self.root, TEST_MANIFEST, Split.TEST, self.category, self.task_type)

    @property
    def categories(self) -> list[str]:
        """全部类别（train_manifest 去重排序），便于逐类循环训练。"""
        self._ensure_manifests()
        df = pd.read_csv(self.root / TRAIN_MANIFEST, encoding="utf-8-sig")
        return sorted(df["category"].astype(str).str.strip().unique().tolist())
