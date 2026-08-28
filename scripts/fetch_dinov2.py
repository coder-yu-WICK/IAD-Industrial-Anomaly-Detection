#!/usr/bin/env python
"""开发用一次性脚本：拉取官方 DINOv2 ViT-L/14 权重、键名对齐、前向一致性校验，
打包到 model/pretrained/dinov2_vitl14.pth 并更新 pretrained_manifest.json。

仅在**联网开发机**（如 Colab）运行一次；生成的 .pth 随提交包走，评测环境断网不依赖本脚本。
本脚本不进入提交 zip。

用法::

    python scripts/fetch_dinov2.py [--entry dinov2_vitl14_reg] [--img-size 518]

注意: torch.hub.load 会下载 dinov2 仓库与权重（~1.1GB）；运行时依赖仍只有
torch/torchvision/numpy/Pillow（requirements.lock 不变）。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from patchcore.dinov2 import build_dinov2_vitl14  # noqa: E402

PRETRAINED = ROOT / "model" / "pretrained" / "dinov2_vitl14.pth"
MANIFEST = ROOT / "pretrained_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-repo", default="facebookresearch/dinov2")
    ap.add_argument("--entry", default="dinov2_vitl14_reg", help="dinov2_vitl14_reg（4 寄存器）或 dinov2_vitl14")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--force", action="store_true", help="强制重新下载/覆盖已存在的权重")
    args = ap.parse_args()

    if PRETRAINED.exists() and not args.force:
        print(f"[skip] 权重已存在 {PRETRAINED}，跳过下载（如需重新下载请加 --force）")
        return

    print(f"[1/5] torch.hub.load('{args.hub_repo}', '{args.entry}')")
    hub_kwargs = {}
    if "trust_repo" in inspect.signature(torch.hub.load).parameters:
        hub_kwargs["trust_repo"] = True  # 新版 torch 需显式信任第三方 hub 仓库
    ref = torch.hub.load(args.hub_repo, args.entry, **hub_kwargs)
    ref.eval()
    ref_sd = ref.state_dict()
    print(f"  主干 {type(ref).__name__}，state_dict {len(ref_sd)} 键")
    for k, v in list(ref_sd.items())[:6]:
        print(f"    {k}: {tuple(v.shape)}")

    n_reg = 0 if getattr(ref, "register_tokens", None) is None else ref.register_tokens.shape[1]
    print(f"  寄存器 token 数: {n_reg}")

    print(f"[2/5] 构建自写 DinoV2(img_size={args.img_size}, registers={n_reg})")
    ours = build_dinov2_vitl14(img_size=args.img_size, num_register_tokens=n_reg)

    missing, unexpected = ours.load_state_dict(dict(ref_sd), strict=False)
    if missing or unexpected:
        print(f"  缺失 {len(missing)} 键: {list(missing)[:8]}")
        print(f"  多余 {len(unexpected)} 键: {list(unexpected)[:8]}")
        raise SystemExit("键名未完全对齐。请对照 dinov2.py 调整（重点：MLP fc1/fc2、LayerScale gamma、寄存器顺序）")
    print("  键名完全对齐 ✓ (0 缺失 / 0 多余)")

    print("[3/5] 前向一致性校验（CPU，规避 GPU flash attention 数值差异）")
    ours = ours.to("cpu")
    ref = ref.to("cpu")
    x = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        out_ref = ref(x)
        out_ours = ours(x)
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    if isinstance(out_ref, dict):
        out_ref = out_ref["x_norm_clstoken"]
    # ref 返回 CLS 向量 (B,D)；ours 返回全 token (B,L,D) → 取 ours 的 CLS 对齐
    ours_cls = out_ours[:, 0] if out_ours.dim() == 3 else out_ours
    d = (ours_cls.float() - out_ref.float()).abs().max().item()
    if d > 1e-4:
        raise SystemExit(
            f"前向不一致 max|diff|={d:.4e} > 1e-4；请对照 dinov2.py "
            "（重点：寄存器顺序 / pos_embed / LayerNorm eps=1e-6 / GELU）"
        )
    print(f"  前向一致 ✓ (max|diff|={d:.2e})")

    print(f"[4/5] 打包到 {PRETRAINED}")
    PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ours.state_dict(),
            "source": f"{args.hub_repo}:{args.entry}",
            "img_size": args.img_size,
            "num_register_tokens": n_reg,
        },
        PRETRAINED,
    )
    sha = hashlib.sha256(PRETRAINED.read_bytes()).hexdigest()
    print(f"  sha256: {sha}")

    print("[5/5] 更新 pretrained_manifest.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "name": "dinov2_vitl14_lvd142m",
        "description": "DINOv2 ViT-L/14 主干在 LVD-142M 上的自监督预训练权重（含 4 寄存器 token）",
        "source_url": f"https://github.com/{args.hub_repo} (torch.hub:{args.entry})",
        "sha256": sha,
        "local_path": "model/pretrained/dinov2_vitl14.pth",
        "license": "Apache-2.0",
        "usage": "src/train.py 默认主干（--backbone dinov2_vitl14），仅作特征提取器，训练阶段冻结",
    }
    manifest["entries"] = [e for e in manifest.get("entries", []) if e.get("name") != "dinov2_vitl14_lvd142m"]
    manifest["entries"].insert(0, entry)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成。请手动核实：")
    print("  1) DINOv2 许可证（Apache-2.0），必要时补 third_party/LICENSES.md")
    print("  2) 将 model/pretrained/dinov2_vitl14.pth 纳入提交包（评测断网依赖此文件）")


if __name__ == "__main__":
    main()
