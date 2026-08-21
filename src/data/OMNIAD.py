# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Omni-AD Data Module（manifest CSV 驱动）。

与 anomalib 内置的 ``MVTecAD`` 继承相同的父类 ``AnomalibDataModule``，因此
在 ``Engine`` / ``Trainer`` 中的用法完全一致，脚本里可以直接把 ``MVTecAD(...)``
换成 ``OMNIAD(...)``。

数据流（manifest 驱动）:
    1. 运行时（prepare_data / setup）检查 ``<root>/train_manifest.csv`` 与
       ``<root>/test_manifest.csv`` 是否存在；
    2. 任一缺失时自动调用同目录 ``sample_manifest.py`` 扫描目录结构生成两个 CSV；
    3. 之后所有操作（train_data / test_data / dataloader）只读 CSV，不再扫描目录。

manifest 列说明（由 ``sample_manifest.py`` 生成）::

    train_manifest.csv   sample_id, category, image_path            # 全部正常样本
    test_manifest.csv    sample_id, category, image_path, anomaly, ground_truth
                          # anomaly=True 为异常样本；ground_truth 为对应掩膜（正常样本为空）

用法::

    >>> from src.data.OMNIAD import OMNIAD
    >>> datamodule = OMNIAD(
    ...     root="./data/Omni-AD-30-release",
    ...     category=None,          # None=加载全部类别，指定则只加载该类别
    ...     train_batch_size=16,
    ...     eval_batch_size=16,
    ...     num_workers=2,
    ... )
    >>> datamodule.setup()
    >>> len(datamodule.train_data), len(datamodule.test_data)
    (2252, 2212)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)

TRAIN_MANIFEST = "train_manifest.csv"
TEST_MANIFEST = "test_manifest.csv"

_ANOMALY_TRUE = {"true", "1", "1.0"}


def make_omniad_samples(
    root: Path | str,
    manifest: str,
    split: str | Split,
    category: str | None = None,
    task: str | TaskType = TaskType.SEGMENTATION,
) -> pd.DataFrame:
    """从 manifest CSV 构建 anomalib samples DataFrame。

    Args:
        root: 数据根目录（manifest 所在目录，其下含各类别目录）。
        manifest: 要读取的 manifest 文件名（``train_manifest.csv`` 或 ``test_manifest.csv``）。
        split: 数据集划分（train/test），写入 DataFrame 的 ``split`` 列。
        category: 指定类别时仅保留该类别；``None`` 表示全部类别。
        task: 任务类型，写入 ``attrs["task"]``，决定 __getitem__ 是否读取掩膜。

    Returns:
        DataFrame，列含 ``image_path``(绝对) / ``split`` / ``label`` /
        ``label_index``(0=正常,1=异常) / ``mask_path``(异常样本的掩膜，正常为空) /
        ``sample_id`` / ``category``，以及 ``attrs["task"]``。

    Raises:
        FileNotFoundError: manifest 文件不存在。
        RuntimeError: 过滤后没有任何样本。
    """
    root = Path(root)
    manifest_path = root / manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest 缺失，请先运行 sample_manifest.py 生成: {manifest_path}"
        )

    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # 类别过滤（None 时保留全部类别）
    if category is not None:
        df = df[df["category"] == category]
    if df.empty:
        raise RuntimeError(f"manifest 中无样本: category={category} manifest={manifest_path}")

    # label_index / label：train manifest 无 anomaly 列，全部正常；
    # test manifest 按 anomaly 列区分 正常(False)/异常(True)。
    if "anomaly" in df.columns:
        anomalous = df["anomaly"].astype(str).str.strip().str.lower().isin(_ANOMALY_TRUE)
        df["label_index"] = anomalous.map({True: LabelName.ABNORMAL, False: LabelName.NORMAL})
        df["label"] = anomalous.map({True: "abnormal", False: "good"})
    else:
        df["label_index"] = LabelName.NORMAL
        df["label"] = "good"

    # image_path 转绝对路径（AnomalibDataset.samples setter 会校验文件存在）
    df["image_path"] = df["image_path"].map(lambda p: str(root / str(p).strip()))

    # mask_path：仅异常 test 样本有 ground_truth，正常样本留空
    if "ground_truth" in df.columns:
        has_gt = df["ground_truth"].notna() & df["ground_truth"].astype(str).str.strip().ne("")
        df["mask_path"] = ""
        df.loc[has_gt, "mask_path"] = df.loc[has_gt, "ground_truth"].map(
            lambda p: str(root / str(p).strip())
        )
    else:
        df["mask_path"] = ""

    df["split"] = split.value if isinstance(split, Split) else split
    df = df.reset_index(drop=True)

    task_str = task.value if isinstance(task, TaskType) else str(task)
    df.attrs["task"] = task_str
    return df


class OMNIADDataset(AnomalibDataset):
    """Omni-AD 数据集，样本信息由 manifest CSV 提供。

    支持 ``classification`` 与 ``segmentation`` 两种任务（任务由 ``task`` 决定，
    默认 segmentation）。为便于调试，DataFrame 额外保留 ``sample_id`` 列。
    """

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
        self.split = Split(split) if isinstance(split, str) else split
        self.samples = make_omniad_samples(
            self.root,
            manifest,
            self.split,
            category=self.category,
            task=task,
        )


class OMNIAD(AnomalibDataModule):
    """Omni-AD Datamodule（manifest 驱动）。

    Args:
        root (Path | str | None): 数据根目录，manifest 与各类别目录所在处。
            默认 ``"./data/Omni-AD-30-release"``（相对仓库根，与 sample_manifest.py 一致）。
        category (str | None): Omni-AD 类别名（如 ``"air_conditioner_filter"``）。
            为 ``None`` 时加载全部类别。默认 ``None``。
        task (str | TaskType): 任务类型，``segmentation``（默认）或 ``classification``。
        train_batch_size (int, optional): 训练 batch size。默认 ``32``。
        eval_batch_size (int, optional): 测试 batch size。默认 ``32``。
        num_workers (int, optional): 数据加载 worker 数。默认 ``8``。
        train_augmentations (Transform | None): 训练集增强。默认 ``None``。
        val_augmentations (Transform | None): 验证集增强。默认 ``None``。
        test_augmentations (Transform | None): 测试集增强。默认 ``None``。
        augmentations (Transform | None): 通用增强（未指定各阶段增强时使用）。
        test_split_mode (TestSplitMode): 测试集切分方式。默认 ``TestSplitMode.FROM_DIR``。
        test_split_ratio (float): 测试集占比。默认 ``0.2``。
        val_split_mode (ValSplitMode): 验证集切分方式。默认 ``ValSplitMode.SAME_AS_TEST``。
        val_split_ratio (float): 验证集占比。默认 ``0.5``。
        seed (int | None, optional): 可复现切分的随机种子。默认 ``None``。
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

        root = root if root is not None else Path(__file__).resolve().parents[2] / "data" / "Omni-AD-30-release"
        self.root = Path(root)
        self.category = category
        # 注意：基类已有只读属性 task（由数据集 attrs["task"] 推出），
        # 这里用 task_type 保存任务配置，避免与属性冲突。
        self.task_type = TaskType(task) if isinstance(task, str) else task

    # ------------------------------------------------------------------ #
    # manifest 生成
    # ------------------------------------------------------------------ #
    def _ensure_manifests(self) -> None:
        """检查 manifest 是否存在，缺失时调用 sample_manifest.py 生成。"""
        needed = [self.root / TRAIN_MANIFEST, self.root / TEST_MANIFEST]
        if all(p.exists() for p in needed):
            return
        if not self.root.is_dir():
            raise NotADirectoryError(f"--data-root 不存在: {self.root}")

        logger.info("[OMNIAD] manifest 缺失，调用 sample_manifest.py 生成 ...")
        script = Path(__file__).resolve().parent / "sample_manifest.py"
        subprocess.run(
            [sys.executable, "-u", str(script), "--data-root", str(self.root)],
            check=True,
        )
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(f"生成 manifest 后仍缺失: {missing}")

    def prepare_data(self) -> None:
        """确保 manifest 存在（缺失时调用 sample_manifest.py 生成）。"""
        self._ensure_manifests()

    # ------------------------------------------------------------------ #
    # 数据集构造（直接读 CSV）
    # ------------------------------------------------------------------ #
    def _setup(self, _stage: str | None = None) -> None:
        """依据 manifest CSV 构建 train / test 数据集。

        Note:
            ``_stage`` 参数不使用：与 MVTecAD 一致，首次 setup 时即创建全部子集，
            以便从测试集切出验证集。
        """
        self._ensure_manifests()
        self.train_data = OMNIADDataset(
            root=self.root,
            manifest=TRAIN_MANIFEST,
            split=Split.TRAIN,
            category=self.category,
            task=self.task_type,
        )
        self.test_data = OMNIADDataset(
            root=self.root,
            manifest=TEST_MANIFEST,
            split=Split.TEST,
            category=self.category,
            task=self.task_type,
        )
