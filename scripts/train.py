import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，保证 src 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lightning.pytorch import seed_everything
from src.data.OMNIAD import OMNIAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

seed_everything(42, workers=True)

# TODO: 拿到 Omni-AD 数据后，替换为实际类别
CATEGORY = "bottle"
BATCH_SIZE = 16
CORESET_RATIO = 0.1
EPOCHS = 1
DEVICE = "auto"
NUM_WORKERS = 2

# 数据加载（使用默认预处理）
datamodule = OMNIAD(
    root="./data",
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

model = Patchcore(
    backbone="vit_base_patch16_224",
    layers=["blocks.10", "blocks.11"],
    coreset_sampling_ratio=CORESET_RATIO,
)

engine = Engine(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    devices=1,
)

print(f"开始训练 PatchCore 模型，类别: {CATEGORY}")
engine.fit(model=model, datamodule=datamodule)
print("训练完成！")