import torch
from lightning.pytorch import seed_everything
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from torchvision import transforms

seed_everything(42, workers=True)

# ==================== 配置参数 ====================
CATEGORY = "bottle"
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
CORESET_RATIO = 0.1
EPOCHS = 1
DEVICE = "auto"

# ==================== 定义数据增强（包含尺寸调整） ====================

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== 数据加载 ====================
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=4,
    train_augmentations=train_transform,
    val_augmentations=eval_transform,
    test_augmentations=eval_transform,
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
print("训练完成！checkpoint 保存在 results/ 目录中。")