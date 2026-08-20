from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

CATEGORY = "bottle"
BATCH_SIZE = 16
DEVICE = "auto"
CKPT_PATH = "./results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt"

# 数据预处理
datamodule = MVTecAD(
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