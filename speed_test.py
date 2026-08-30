import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.patchcore import PatchCore

def parse_args():
    parser = argparse.ArgumentParser(description="Test single-card inference speed for PatchCore")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g., cuda:0 or cpu)")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of iterations for speed test")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--image-size", type=int, default=518, help="Dummy image size (square)")
    parser.add_argument("--model-dir", type=str, default="work/model", help="Directory containing shared.pth and checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Use default backbone and layers matching train.py
    model = PatchCore(
        device=device,
        backbone="dinov2_vitl14",
        layers=("blocks.6", "blocks.12", "blocks.18"),
    )

    model_dir = Path(args.model_dir)
    shared_path = model_dir / "shared.pth"
    if not shared_path.exists():
        print(f"Error: {shared_path} not found.")
        return
        
    print(f"Loading shared backbone from {shared_path}...")
    model.load_shared(shared_path)

    # Load first available checkpoint
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"Error: {ckpt_dir} directory not found.")
        return
        
    checkpoints = list(ckpt_dir.glob("*.pth"))
    if not checkpoints:
        print(f"Error: No checkpoints found in {ckpt_dir}.")
        return
        
    ckpt_path = checkpoints[0]
    print(f"Loading category bank from {ckpt_path}...")
    model.load_category(ckpt_path)

    # Try ONNX fallback
    onnx_path = model_dir / "shared.onnx"
    if onnx_path.exists():
        try:
            model.load_onnx(onnx_path)
            print(f"Using ONNX acceleration from {onnx_path}")
        except Exception as e:
            print(f"ONNX load failed: {e}")
    else:
        print("No ONNX model found, using PyTorch inference.")

    # Create dummy image
    print(f"Generating dummy image of size {args.image_size}x{args.image_size}...")
    dummy_np = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy_np).convert("RGB")

    print(f"Warming up for {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            model.predict(dummy_img)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Running speed test for {args.num_iters} iterations...")
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(args.num_iters):
            model.predict(dummy_img)

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    latency = (total_time / args.num_iters) * 1000
    fps = args.num_iters / total_time

    print("\n" + "=" * 40)
    print("Inference Speed Test Results")
    print("=" * 40)
    print(f"Device      : {args.device}")
    print(f"Image Size  : {args.image_size}x{args.image_size}")
    print(f"Iterations  : {args.num_iters}")
    print(f"Total Time  : {total_time:.4f} seconds")
    print(f"Latency     : {latency:.2f} ms / image")
    print(f"Throughput  : {fps:.2f} FPS")
    print("=" * 40)

if __name__ == "__main__":
    main()
