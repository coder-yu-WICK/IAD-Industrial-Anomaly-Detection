"""
将训练好的 PatchCore 模型导出为 ONNX 格式
"""

from pathlib import Path
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# config
CATEGORY = "bottle"
CKPT_PATH = "./results/Patchcore/MVTecAD/bottle/v2/weights/lightning/model.ckpt"
EXPORT_ROOT = "./results/exported"
EXPORT_TYPE = "onnx"

model = Patchcore(
    backbone="vit_base_patch16_224",
    layers=["blocks.10", "blocks.11"],
    coreset_sampling_ratio=0.1,
)

datamodule = MVTecAD(
    category=CATEGORY,
    eval_batch_size=1,
    num_workers=0,
)
datamodule.setup()

engine = Engine(accelerator="auto", devices=1)

engine.export(
    model=model,
    ckpt_path=CKPT_PATH,
    export_type=EXPORT_TYPE,
    export_root=EXPORT_ROOT,
    datamodule=datamodule,
)

print(f"Export model to: {EXPORT_ROOT}/{EXPORT_TYPE}/")