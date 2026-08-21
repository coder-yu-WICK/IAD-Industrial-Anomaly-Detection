import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，保证 src 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.OMNIAD import OMNIAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# TODO: 拿到 Omni-AD 数据后，替换为实际类别
CATEGORY = "bottle"
CKPT_PATH = "./results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt"
OUTPUT_DIR = "./results/visulization"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 20

datamodule = OMNIAD(
    category=CATEGORY,
    eval_batch_size=1,
    num_workers=0,
)
datamodule.setup()

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

engine = Engine(accelerator="auto", devices=1)

print(f"Start inference with category: {CATEGORY}")

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,
    return_predictions=True,
)

test_dataset = datamodule.test_data

for idx, pred in enumerate(predictions[:MAX_SAMPLES]):
    sample = test_dataset[idx]
    mask_tensor = sample.gt_mask

    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor.squeeze(0)
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor.squeeze(0)
    gt_mask = mask_tensor.cpu().numpy().astype(np.float32)  # (H, W)

    img_tensor = pred["image"]
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img_tensor.cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1) * 255
    img_uint8 = img_np.astype(np.uint8)

    amap_tensor = pred["anomaly_map"]
    if amap_tensor.ndim == 4:
        amap_tensor = amap_tensor.squeeze(0)
    if amap_tensor.ndim == 3 and amap_tensor.shape[0] == 1:
        amap_tensor = amap_tensor.squeeze(0)
    amap = amap_tensor.cpu().numpy()   # (H, W)

    score = pred["pred_score"].item()
    label = "Anomalous" if pred["pred_label"].item() else "Normal"

    # 生成二值化预测掩码（阈值0.5
    pred_mask = (amap > 0.5).astype(np.float32)

    # 叠加图：原图 + 预测掩码（半透明红色）
    overlay = img_uint8.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255
    alpha = 0.4
    mask_indices = pred_mask > 0
    overlay[mask_indices] = (1 - alpha) * overlay[mask_indices] + alpha * red_mask[mask_indices]
    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(img_uint8)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(amap, cmap="jet")
    axes[2].set_title(f"Anomaly Map\nScore: {score:.4f} ({label})")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Pred Mask on Image")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/result_{idx:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"已保存第 {idx+1} 张: {OUTPUT_DIR}/result_{idx:03d}.png")
