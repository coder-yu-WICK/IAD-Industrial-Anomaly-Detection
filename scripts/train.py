import torch
from lightning.pytorch import seed_everything
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.utils.config import update_config

"""
使用 MVTec AD 数据集测试 PatchCore 训练过程
"""

seed_everything(42, workers=True)

# ==================== config ====================
CATEGORY = "bottle"          # MVTec AD 类别，可选 "all" 或具体类别
BATCH_SIZE = 8               # 根据 GPU 显存调整
IMAGE_SIZE = (256, 256)      # 输入图像尺寸
CORESET_RATIO = 0.1          # 核心集采样比例
EPOCHS = 1                   # PatchCore 只需训练 1 个 epoch（仅构建记忆库）
DEVICE = "auto"              # 自动选择设备

# ==================== dataloader ====================
datamodule = MVTecAD(
    category=CATEGORY,
    image_size=IMAGE_SIZE,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=4,
)

# ==================== 模型构建 ====================
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=CORESET_RATIO,
)

# ==================== 训练引擎 ====================
engine = Engine(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    devices=1,
)

# ==================== 开始训练 ====================
print(f"开始训练 PatchCore 模型，类别: {CATEGORY}")
engine.fit(model=model, datamodule=datamodule)

# 训练结束后，模型 checkpoint 会自动保存在 ./results/patchcore/mvtecad/{category}/ 目录下
print("训练完成！checkpoint 保存在 results/ 目录中。")