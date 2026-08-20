import torch
from lightning.pytorch import seed_everything
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

seed_everything(42, workers=True)

CATEGORY = "bottle"
BATCH_SIZE = 16
CORESET_RATIO = 0.1
EPOCHS = 1
DEVICE = "auto"
NUM_WORKERS = 2

# 数据加载（使用默认预处理）
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
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