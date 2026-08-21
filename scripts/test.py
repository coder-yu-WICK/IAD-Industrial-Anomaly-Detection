import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，保证 src 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.OMNIAD import OMNIAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

# TODO: 拿到 Omni-AD 数据后，替换为实际类别
CATEGORY = "bottle"
BATCH_SIZE = 16
DEVICE = "auto"
CKPT_PATH = "./results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt"

# 数据预处理
datamodule = OMNIAD(
    category=CATEGORY,
    eval_batch_size=BATCH_SIZE,
    num_workers=2,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(accelerator=DEVICE, devices=1)

print(f"开始测试，类别: {CATEGORY}")
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,
)

print("\n========== 测试结果 ==========")
for key, value in test_results[0].items():
    print(f"{key}: {value:.4f}")