import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.patchcore import PatchCore
from src.patchcore.anomaly_map import AnomalyMapGenerator
from src.patchcore.preprocess import PatchPreprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Test single-card inference speed for PatchCore")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g., cuda:0 or cpu)")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of iterations for speed test")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--image-size", type=int, default=518, help="Dummy image size (square)")
    parser.add_argument("--model-dir", type=str, default="work/model", help="Directory containing shared.pth and checkpoints")
    parser.add_argument("--data-root", type=str, default="src/data/Omni-AD-30-release",
                        help="Path to dataset root to randomly pick an image from")
    parser.add_argument("--inference-dtype", type=str, default=None,
                        help="Inference precision (None, float32, float16, bfloat16)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable component-level profiling to find bottlenecks")
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
        inference_dtype=args.inference_dtype,
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

    # Load random image or create dummy image
    data_root = Path(args.data_root)
    if data_root.exists() and data_root.is_dir():
        import random
        print(f"Searching for images in {data_root}...")
        image_paths = list(data_root.rglob("*.png")) + list(data_root.rglob("*.jpg"))
        if not image_paths:
            print(f"No images found in {data_root}, falling back to dummy image.")
            print(f"Generating dummy image of size {args.image_size}x{args.image_size}...")
            dummy_np = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
            test_img = Image.fromarray(dummy_np).convert("RGB")
        else:
            chosen_path = random.choice(image_paths)
            print(f"Randomly selected image: {chosen_path}")
            test_img = Image.open(chosen_path).convert("RGB")
            args.image_size = f"{test_img.width}x{test_img.height}" # update for printing later
    else:
        print(f"Data root {data_root} not found.")
        print(f"Generating dummy image of size {args.image_size}x{args.image_size}...")
        dummy_np = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
        test_img = Image.fromarray(dummy_np).convert("RGB")

    print(f"Warming up for {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            model.predict(test_img)

    if device.type == "cuda":
        torch.cuda.synchronize()

    if args.profile:
        print(f"Running component-level profiling for {args.num_iters} iterations...")
        time_pre = 0.0
        time_ext = 0.0
        time_knn = 0.0
        time_post = 0.0

        # setup internal components just like model.predict()
        preprocess = PatchPreprocess.from_dict(model.bank_dict)
        map_gen = AnomalyMapGenerator(sigma=model.bank_dict.get("sigma") or 4.0)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.num_iters):
                # 1. Preprocess
                t0 = time.perf_counter()
                x = preprocess.encode(test_img).to(model.device)
                if device.type == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()

                # 2. Extract
                if model.onnx is not None:
                    x_np = preprocess.encode(test_img).numpy()[None].astype("float32")
                    named = {name: torch.from_numpy(f).to(model.device) for name, f in zip(model.layers, model.onnx(x_np))}
                    patch = model.extractor.aggregate(named)
                else:
                    patch = model.extractor(x)
                if device.type == "cuda": torch.cuda.synchronize()
                t2 = time.perf_counter()

                # 3. k-NN Search
                h, w = patch.shape[-2:]
                q = patch.reshape(patch.shape[0], -1).T
                dist = model._bank.nearest_dist(q)
                if device.type == "cuda": torch.cuda.synchronize()
                t3 = time.perf_counter()

                # 4. Postprocess
                score_map = dist.reshape(1, h, w)
                score_map = map_gen(score_map, test_img.size[::-1])
                score_map = score_map[0]
                score_map = 1.0 - torch.exp(-score_map / model._bank.norm_scale)
                anomaly_map = score_map.cpu().numpy().astype(np.float32)
                image_score = float(anomaly_map.max())
                if device.type == "cuda": torch.cuda.synchronize()
                t4 = time.perf_counter()

                time_pre += (t1 - t0)
                time_ext += (t2 - t1)
                time_knn += (t3 - t2)
                time_post += (t4 - t3)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

        lat_pre = (time_pre / args.num_iters) * 1000
        lat_ext = (time_ext / args.num_iters) * 1000
        lat_knn = (time_knn / args.num_iters) * 1000
        lat_post = (time_post / args.num_iters) * 1000
        lat_total = (total_time / args.num_iters) * 1000
        fps = args.num_iters / total_time

        print("\n" + "=" * 50)
        print("Inference Profiling Results")
        print("=" * 50)
        print(f"Device      : {args.device}")
        print(f"Precision   : {args.inference_dtype or 'CUDA Auto (FP16/FP32)'}")
        print(f"Image Size  : {args.image_size}")
        print(f"Iterations  : {args.num_iters}")
        print("-" * 50)
        print(f"Pre-process : {lat_pre:>6.2f} ms ({time_pre/total_time*100:>5.1f}%)")
        print(f"Extraction  : {lat_ext:>6.2f} ms ({time_ext/total_time*100:>5.1f}%)")
        print(f"k-NN Search : {lat_knn:>6.2f} ms ({time_knn/total_time*100:>5.1f}%)")
        print(f"Post-process: {lat_post:>6.2f} ms ({time_post/total_time*100:>5.1f}%)")
        print("-" * 50)
        print(f"Total Time  : {total_time:.4f} seconds")
        print(f"Avg Latency : {lat_total:.2f} ms / image")
        print(f"Throughput  : {fps:.2f} FPS")
        print("=" * 50)
    else:
        print(f"Running speed test for {args.num_iters} iterations...")
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(args.num_iters):
                model.predict(test_img)

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
        print(f"Precision   : {args.inference_dtype or 'CUDA Auto (FP16/FP32)'}")
        print(f"Image Size  : {args.image_size}")
        print(f"Iterations  : {args.num_iters}")
        print(f"Total Time  : {total_time:.4f} seconds")
        print(f"Latency     : {latency:.2f} ms / image")
        print(f"Throughput  : {fps:.2f} FPS")
        print("=" * 40)

if __name__ == "__main__":
    main()
