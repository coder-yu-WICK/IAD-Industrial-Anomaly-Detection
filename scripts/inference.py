import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from anomalib.deploy import TorchInferencer
from anomalib.data import MVTecAD
from anomalib.utils.config import update_config

# ==================== config ====================
CATEGORY = "bottle"
CONFIG_PATH = f"./results/patchcore/mvtecad/{CATEGORY}/config.yaml"   # 自动生成的配置文件
CKPT_PATH = f"./results/patchcore/mvtecad/{CATEGORY}/weights/epoch_0.ckpt"  # 根据实际路径调整

# test image path
IMAGE_PATH = "path/to/your/test_image.png"

inferencer = TorchInferencer(
    config=CONFIG_PATH,
    model_source=CKPT_PATH,
    device="auto",
)

predictions = inferencer.predict(image=IMAGE_PATH)

image = predictions["image"]
if isinstance(image, torch.Tensor):
    image = image.cpu().numpy()

anomaly_map = predictions["anomaly_map"]
if isinstance(anomaly_map, torch.Tensor):
    anomaly_map = anomaly_map.cpu().numpy()

score = predictions["pred_score"]
label = "Anomalous" if predictions["pred_label"] else "Normal"

image_uint8 = (image * 255).astype(np.uint8)
anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)

heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image_uint8, 0.7, heatmap, 0.3, 0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image_uint8)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(anomaly_map, cmap="jet")
axes[1].set_title(f"Anomaly Map\nScore: {score:.4f} ({label})")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("inference_result.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"推理结果已保存为 inference_result.png")