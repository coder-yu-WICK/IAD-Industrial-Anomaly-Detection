# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""Omni-AD Data Module.

竞赛用 Omni-AD（CVPR 2026）数据的自定义 DataLoader。

该类与 anomalib 内置的 ``MVTecAD`` 继承相同的父类
``AnomalibDataModule``，因此在 ``Engine`` / ``Trainer`` 中的
用法完全一致，脚本里可以直接把 ``MVTecAD(...)`` 换成 ``OMNIAD(...)``。

注意：
    目前数据集尚未到位，具体的加载流程留白（见 ``_setup`` / ``prepare_data``
    中的 TODO）。待组委会提供数据后，按 README 中的约定放置到 ``data/`` 目录，
    再补全这两个方法即可：
"""

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class OMNIAD(AnomalibDataModule):
    """Omni-AD Datamodule。

    Args:
        root (Path | str | None): 数据集根目录。默认 ``"./data/Omni-AD"``
            （组委会数据的统一放置位置，已 gitignore）。
        category (str | None): Omni-AD 类别名（拿到数据后按实际情况填写）。
            为 ``None`` 时表示加载全部类别。默认为 ``None``。
        train_batch_size (int, optional): 训练 batch size。默认 ``32``。
        eval_batch_size (int, optional): 测试 batch size。默认 ``32``。
        num_workers (int, optional): 数据加载 worker 数。默认 ``8``。
        train_augmentations (Transform | None): 训练集增强。默认 ``None``。
        val_augmentations (Transform | None): 验证集增强。默认 ``None``。
        test_augmentations (Transform | None): 测试集增强。默认 ``None``。
        augmentations (Transform | None): 通用增强（未指定各阶段增强时使用）。
        test_split_mode (TestSplitMode): 测试集切分方式。
            默认 ``TestSplitMode.FROM_DIR``。
        test_split_ratio (float): 测试集占比。默认 ``0.2``。
        val_split_mode (ValSplitMode): 验证集切分方式。
            默认 ``ValSplitMode.SAME_AS_TEST``。
        val_split_ratio (float): 验证集占比。默认 ``0.5``。
        seed (int | None, optional): 可复现切分的随机种子。默认 ``None``。

    Example:
        用法与 ``MVTecAD`` 完全一致::

            >>> from src.data.OMNIAD import OMNIAD
            >>> datamodule = OMNIAD(
            ...     root="./data",
            ...     category="bottle",  # TODO: 换成 Omni-AD 实际类别
            ...     train_batch_size=16,
            ...     eval_batch_size=16,
            ...     num_workers=2,
            ... )
    """

    def __init__(
        self,
        root: Path | str | None = "./data",
        category: str | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
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

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """构建训练 / 测试数据集。

        TODO: 数据集尚未到位，具体加载流程留白。拿到 Omni-AD 数据后，
        参照 ``MVTecAD._setup`` 的写法，在此创建并赋值：:

            self.train_data = ...
            self.test_data = ...
            # 如需验证集，可自行创建 self.val_data
        """
        # self.train_data = OMNIADDataset(split=Split.TRAIN, root=self.root, category=self.category)
        # self.test_data = OMNIADDataset(split=Split.TEST, root=self.root, category=self.category)
        pass

    def prepare_data(self) -> None:
        """数据下载 / 预处理（可选）。

        TODO: 数据由组委会提供，无需下载。如官方压缩包需解压 / 转换格式，
        可在此实现，例如 ``download_and_extract(self.root, DOWNLOAD_INFO)``。
        """
        pass
