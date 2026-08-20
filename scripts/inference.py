"""
单张推理时间测试 (ONNX Runtime)
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import onnxruntime as ort

# config
ONNX_PATH = "./results/exported/weights/onnx/model.onnx"
TEST_IMAGE_PATH = "./datasets/MVTecAD/bottle/test/broken_large/000.png"
WARMUP_RUNS = 5
TEST_RUNS = 50
IMAGE_SIZE = (256, 256)

preprocess = Compose([
    Resize(IMAGE_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return preprocess(img).unsqueeze(0).numpy()  # (1, 3, H, W)

session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
print(f"输入名称: {input_name}")

input_tensor = load_image(TEST_IMAGE_PATH)

print("预热 ONNX Runtime...")
for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: input_tensor})

print(f"开始测试 (共 {TEST_RUNS} 次)...")
times = []
for i in range(TEST_RUNS):
    start = time.perf_counter()
    session.run(None, {input_name: input_tensor})
    end = time.perf_counter()
    times.append((end - start) * 1000)
    if (i+1) % 10 == 0:
        print(f"  已测 {i+1}/{TEST_RUNS} 次")

avg = np.mean(times)
std = np.std(times)
min_t = np.min(times)
max_t = np.max(times)

print("\n========== Result ==========")
print(f"平均耗时: {avg:.2f} ms")
print(f"标准差:  {std:.2f} ms")
print(f"最小值:  {min_t:.2f} ms")
print(f"最大值:  {max_t:.2f} ms")