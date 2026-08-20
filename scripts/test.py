# test.py
"""
加载训练好的 PatchCore 模型，在 MVTec AD 测试集上进行评估，并打印指标。
"""

import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.utils.config import update_config
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

# ==================== config ====================
CATEGORY = "bottle"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
DEVICE = "auto"

# 指定训练好的 checkpoint 路径
CKPT_PATH = None  # 例如 "./results/patchcore/mvtecad/bottle/weights/epoch_0.ckpt"

# ==================== dataloader ====================
datamodule = MVTecAD(
    category=CATEGORY,
    image_size=IMAGE_SIZE,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=4,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,   # 必须与训练时一致
)

engine = Engine(accelerator=DEVICE, devices=1)

# ==================== test ====================
print(f"开始测试，类别: {CATEGORY}")
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,   # 若为 None，自动加载最新 checkpoint
)

print("\n========== 测试结果 ==========")
for key, value in test_results[0].items():
    print(f"{key}: {value:.4f}")