# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""推理/训练耗时与显存基准：对照评测硬约束（单张推理 ≤100ms，峰值显存 ≤24GB）。

用法::

    python -u src/benchmark.py --mode both --model-dir work/model \
        --data-root src/data/Omni-AD-30-release --device cuda:0

    python -u src/benchmark.py --mode infer --synthetic --device cuda:0

模式:
    infer   用 --model-dir 里的真实 bank 对测试图分段计时（preprocess/主干/级联检索/热图）。
    train   用 Franca 主干逐类跑 fit() 计时（总耗时÷类别数 计分项）。
    both    先 train 后 infer（同一次进程）。
    synthetic 无模型产物时用随机图 + 内存临时 bank 冒烟，仅验证脚本与管线，数值非真实规模。

阶段计时：CUDA 上用 torch.cuda.Event，CPU 上退化为 perf_counter；统计 mean/p50/p95/p99。
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from patchcore import PatchCore, PatchPreprocess, AnomalyMapGenerator
from patchcore.cascade import CascadeMemoryBank, build_bank

INFER_BUDGET_MS = 100.0
MEM_BUDGET_GB = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchCore 推理/训练耗时显存基准")
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--data-root", type=Path, default=Path("src/data/Omni-AD-30-release"))
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=60)
    parser.add_argument("--num-categories", type=int, default=3, help="train 模式：计时类别数，0=全部")
    parser.add_argument("--profile", type=int, default=0, help="对前 N 张跑 torch.profiler 输出算子归因")
    parser.add_argument("--synthetic", action="store_true", help="用随机图 + 内存临时 bank 冒烟")
    parser.add_argument("--synthetic-fit", type=int, default=8, help="synthetic 模式临时 bank 拟合张数")
    parser.add_argument("--synthetic-size", type=str, default="1000 800")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def groups_of(rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["category"], []).append(r["image_path"])
    return out


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


class StageTimer:
    """分段计时：CUDA 用 Event，CPU 用 perf_counter。"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = is_cuda(device)
        self._marks: dict[str, object] = {}
        self._start_cpu: dict[str, float] = {}

    def mark(self, name: str) -> None:
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._marks[name] = ev
        else:
            self._start_cpu[name] = time.perf_counter()

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    def elapsed(self, a: str, b: str) -> float:
        if self.cuda:
            return self._marks[a].elapsed_time(self._marks[b])
        return (self._start_cpu[b] - self._start_cpu[a]) * 1000.0


def infer_once(model: PatchCore, img: Image.Image, device: torch.device, timer: StageTimer) -> dict[str, float]:
    """复刻 model.predict 内部步骤并分段计时，返回 {stage: ms} 与总 ms。"""
    bd = model.bank_dict
    if model._bank is None:
        model._bank = build_bank(bd, device=device)
    bank = model._bank
    preprocess = PatchPreprocess.from_dict(bd)
    map_gen = AnomalyMapGenerator(sigma=bd.get("sigma") or 4.0)

    if timer.cuda:
        torch.cuda.synchronize()
    timer.mark("preprocess")
    x = preprocess.encode(img).to(device)
    timer.sync()
    timer.mark("extract")
    feats = model.extractor(x)
    timer.mark("retrieve")
    if isinstance(bank, CascadeMemoryBank):
        query = {lv: f.reshape(f.shape[0], -1).T for lv, f in feats.items()}
        h, w = feats[model.layers[0]].shape[-2:]
        dist = bank(query)
    else:
        patch = model.extractor.concat_feature(x)
        h, w = patch.shape[-2:]
        dist = bank.nearest_dist(patch.reshape(patch.shape[0], -1).T)
    timer.mark("map")
    score_map = dist.reshape(1, h, w)
    score_map = map_gen(score_map, img.size[::-1])[0]
    score_map = 1.0 - torch.exp(-score_map / bank.norm_scale)
    score_map.cpu()
    timer.mark("done")
    timer.sync()

    pre_ms = timer.elapsed("preprocess", "extract")
    ex_ms = timer.elapsed("extract", "retrieve")
    rt_ms = timer.elapsed("retrieve", "map")
    mp_ms = timer.elapsed("map", "done")
    return {"preprocess": pre_ms, "extract": ex_ms, "retrieve": rt_ms, "map": mp_ms, "total": pre_ms + ex_ms + rt_ms + mp_ms}


def stats(vals: list[float]) -> tuple[float, float, float, float]:
    v = sorted(vals)
    n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return statistics.mean(v), q(0.50), q(0.95), q(0.99)


def print_row(label: str, m, p50, p95, p99) -> None:
    print(f"  {label:<12} mean={m:8.2f}ms  p50={p50:8.2f}  p95={p95:8.2f}  p99={p99:8.2f}")


def print_verdict(tag: str, ok: bool) -> None:
    print(f"  [判定] {tag}: {'PASS' if ok else 'FAIL / 超标'}")


def sample_rows(rows: list[dict], per_category: int, cap: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    picked: list[dict] = []
    for cat, items in sorted(by_cat.items()):
        picked.extend(random.sample(items, min(per_category, len(items))))
        if len(picked) >= cap:
            break
    return picked[:cap]


def make_synthetic(size: tuple[int, int]) -> Image.Image:
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def load_banks(model_dir: Path) -> list[str]:
    """校验 --model-dir 下类别 bank 均为新格式（format_version=2），返回类别名列表。"""
    shared = model_dir / "shared.pth"
    ckpt_dir = model_dir / "checkpoints"
    if not shared.exists():
        raise FileNotFoundError(f"缺少 {shared}（需先运行 src/train.py）")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"缺少 {ckpt_dir}/（需先运行 src/train.py）")
    ckpts = sorted(ckpt_dir.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"{ckpt_dir} 下无类别 bank（需先运行 src/train.py）")
    cats: list[str] = []
    for p in ckpts:
        data = torch.load(p, map_location="cpu", weights_only=False)
        bd = data["bank_dict"]
        if "banks" not in bd:
            raise ValueError(
                f"{p} 是旧格式单 bank（无 'banks' 字段），与当前 matryoshka 分支不兼容；"
                "请用 src/train.py 重新训练"
            )
        cats.append(data["category"])
    return cats


def run_infer(args, device: torch.device, categories: list[str], shared_path: Path, rows: list[dict]) -> None:
    print(f"\n==== 推理基准（{len(categories)} 个类别 bank）====")
    model = PatchCore(device=device, backbone="franca_vitb14", layers=("blocks.3", "blocks.6", "blocks.9"))
    model.load_shared(shared_path)

    samples = sample_rows(rows, args.per_category, args.num_images)
    if not samples:
        raise RuntimeError("eval manifest 为空")

    imgs: list[tuple[str, Image.Image]] = []
    for s in samples:
        if s["category"] not in categories:
            raise FileNotFoundError(f"eval 中类别 {s['category']} 没有对应 bank（{args.model_dir}/checkpoints）")
        imgs.append((s["category"], Image.open(args.data_root / s["image_path"]).convert("RGB")))

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_stage: dict[str, list[float]] = {k: [] for k in ("preprocess", "extract", "retrieve", "map", "total")}
    per_cat: dict[str, list[float]] = {}
    timer = StageTimer(device)
    cur_cat = None
    for cat, img in imgs:
        if cat != cur_cat:
            model.load_category(Path(args.model_dir) / "checkpoints" / f"{cat}.pth")
            cur_cat = cat
            for _ in range(args.warmup):
                infer_once(model, img, device, timer)
        res = infer_once(model, img, device, timer)
        for k, v in res.items():
            per_stage[k].append(v)
        per_cat.setdefault(cat, []).append(res["total"])

    print("--- 分段耗时（所有被测图）---")
    for k in ("preprocess", "extract", "retrieve", "map", "total"):
        print_row(k, *stats(per_stage[k]))
    print("--- 逐类别端到端 ---")
    for cat, vals in sorted(per_cat.items()):
        print_row(cat, *stats(vals))

    t_mean, _, t_p95, _ = stats(per_stage["total"])
    print_verdict(f"推理平均 {t_mean:.2f}ms <= 100ms", t_mean <= INFER_BUDGET_MS)
    print_verdict(f"推理 p95 {t_p95:.2f}ms <= 100ms", t_p95 <= INFER_BUDGET_MS)

    if torch.cuda.is_available() and device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        print(f"  推理峰值显存: {peak:.2f} GB")
        print_verdict(f"峰值显存 {peak:.2f}GB <= 24GB", peak <= MEM_BUDGET_GB)

    if args.profile > 0:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            cur = None
            for cat, img in imgs[: args.profile]:
                if cat != cur:
                    model.load_category(Path(args.model_dir) / "checkpoints" / f"{cat}.pth")
                    cur = cat
                infer_once(model, img, device, StageTimer(device))
        print("--- 算子归因（CUDA time top）---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def run_train(args, device: torch.device, train_rows: list[dict], pretrained: Path) -> None:
    from train import BACKBONE, LAYERS, CASCADE_RATIOS, MAX_EMBED

    by_cat = groups_of(train_rows)
    cats = sorted(by_cat.keys())
    if args.num_categories > 0:
        cats = cats[: args.num_categories]

    print(f"\n==== 训练基准（{len(cats)} 个类别，Franca {BACKBONE}）====")
    model = PatchCore(
        device=device,
        backbone=BACKBONE,
        layers=LAYERS,
        cascade_ratios=CASCADE_RATIOS,
        max_embed=MAX_EMBED,
        pretrained_path=pretrained,
    )

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    totals: list[float] = []
    for cat in cats:
        paths = [args.data_root / p for p in by_cat[cat]]
        t0 = time.perf_counter()
        model.fit(paths)
        if is_cuda(device):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        totals.append(dt)
        print(f"  {cat:<24} samples={len(paths):<4} fit={dt/1000:.2f}s")

    tot = sum(totals) / 1000.0
    print(f"  {len(cats)} 类总耗时 {tot:.2f}s，平均每类 {tot/len(cats):.2f}s")

    if torch.cuda.is_available() and device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        print(f"  训练峰值显存: {peak:.2f} GB")
        print_verdict(f"训练峰值显存 {peak:.2f}GB <= 24GB", peak <= MEM_BUDGET_GB)


def run_synthetic(args, device: torch.device) -> None:
    from train import BACKBONE, LAYERS, CASCADE_RATIOS, MAX_EMBED, get_pretrained

    print("\n==== synthetic 冒烟（随机图 + 内存临时 bank，数值非真实规模）====")
    size = tuple(int(v) for v in args.synthetic_size.split())
    model = PatchCore(
        device=device,
        backbone=BACKBONE,
        layers=LAYERS,
        cascade_ratios=CASCADE_RATIOS,
        max_embed=MAX_EMBED,
        pretrained_path=get_pretrained(),
    )
    imgs = [make_synthetic(size) for _ in range(args.synthetic_fit)]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(imgs):
            p = Path(tmp) / f"{i:03d}.png"
            img.save(p)
            paths.append(p)
        model.fit(paths)

    bank = build_bank(model.bank_dict, device=device)
    model._bank = bank
    for _ in range(args.warmup):
        infer_once(model, imgs[0], device, StageTimer(device))

    per_stage: dict[str, list[float]] = {k: [] for k in ("preprocess", "extract", "retrieve", "map", "total")}
    timer = StageTimer(device)
    for img in imgs:
        res = infer_once(model, img, device, timer)
        for k, v in res.items():
            per_stage[k].append(v)

    print("--- 分段耗时 ---")
    for k in ("preprocess", "extract", "retrieve", "map", "total"):
        print_row(k, *stats(per_stage[k]))
    print(f"  临时 bank 行数: {bank.banks[model.layers[0]].shape[0]}（真实 coreset 后约 5000 行/层）")


def main() -> None:
    args = parse_args()
    set_seed(2026)
    device = torch.device(args.device)
    pretrained = Path("model/pretrained/franca_vitb14.pth")
    data_root = args.data_root
    train_manifest = args.train_manifest or data_root / "train_manifest.csv"
    eval_manifest = args.eval_manifest or data_root / "test_manifest.csv"

    train_rows = read_manifest(train_manifest)
    eval_rows = read_manifest(eval_manifest)

    if args.synthetic:
        run_synthetic(args, device)
        return

    if args.mode in ("train", "both"):
        run_train(args, device, train_rows, pretrained)

    if args.mode in ("infer", "both"):
        categories = load_banks(args.model_dir)
        run_infer(args, device, categories, args.model_dir / "shared.pth", eval_rows)


if __name__ == "__main__":
    main()
