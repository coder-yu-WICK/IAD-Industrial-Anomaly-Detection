#!/usr/bin/env python
"""开发用一次性脚本：拉取 Franca ViT-B/14 权重、对齐键名、前向一致性校验，
打包到 model/pretrained/franca_vitb14.pth 并更新 pretrained_manifest.json。

仅在**联网开发机**运行一次；生成的 .pth 随提交包走，评测环境断网不依赖本脚本。
本脚本不进入提交 zip（开发调试路径，同 scripts/train.py）。

用法::
    python scripts/fetch_franca.py [--hub-repo valeaoi/Franca] [--entry franca_vitb14]

注意: Franca 仓库 hub 入口可能 import timm 等开发期依赖，仅在一次性临时 venv 里
安装即可，不影响 requirements.lock（运行时仅 torch/torchvision/numpy/Pillow）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from patchcore.vit import build_franca_vitb14  # noqa: E402

PRETRAINED = ROOT / "model" / "pretrained" / "franca_vitb14.pth"
MANIFEST = ROOT / "pretrained_manifest.json"


def resolve_backbone(obj: torch.nn.Module) -> torch.nn.Module:
    """hub 返回的可能是包装器，递归找含 ``blocks`` 的 ViT 主干。"""
    if hasattr(obj, "blocks"):
        return obj
    for name in ("backbone", "model", "encoder", "student", "teacher"):
        sub = getattr(obj, name, None)
        if sub is not None:
            return resolve_backbone(sub)
    return obj


def strip_prefix(sd: dict, prefixes=("backbone.", "model.", "encoder.", "student.")) -> dict:
    """若所有键共享某前缀则剥离（Franca 命名空间 → 我们的命名空间）。"""
    for pre in prefixes:
        if all(k.startswith(pre) for k in sd):
            return {k[len(pre):]: v for k, v in sd.items()}
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-repo", default="valeoai/Franca")
    ap.add_argument("--entry", default="franca_vitb14")
    ap.add_argument("--img-size", type=int, default=518)
    args = ap.parse_args()

    print(f"[1/5] torch.hub.load('{args.hub_repo}', '{args.entry}', use_rasa_head=False)")
    ref = resolve_backbone(torch.hub.load(args.hub_repo, args.entry, use_rasa_head=False))
    ref.eval()
    ref_sd = ref.state_dict()
    print(f"  主干 {type(ref).__name__}，state_dict {len(ref_sd)} 键")
    for k, v in list(ref_sd.items())[:6]:
        print(f"    {k}: {tuple(v.shape)}")

    n_reg = 1 if any("register" in k or "reg_token" in k for k in ref_sd) else 0
    print(f"[2/5] 构建自写 DinoViT(img_size={args.img_size}, num_register_tokens={n_reg})")
    ours = build_franca_vitb14(img_size=args.img_size, num_register_tokens=n_reg)

    sd = strip_prefix(dict(ref_sd))
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    if missing:
        print(f"  缺失键（ours 有、ref 无，可能需 --img-size/register 调整）: {list(missing)[:10]}")
    if unexpected:
        print(f"  多余键（ref 有、ours 无，多为 ls/头等，可忽略）: {list(unexpected)[:10]}")
    if missing and not unexpected:
        raise SystemExit(f"键名对齐失败：缺失 {len(missing)} 个键，无法继续。请对照打印的 ref 键名调整 vit.py。")

    print("[3/5] 前向一致性校验（黄金标准）：randn 输入，ours 与 ref 输出 allclose(atol=1e-4)")
    x = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        out_ref = ref(x)
        out_ours = ours(x)
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    if isinstance(out_ref, dict):
        out_ref = out_ref["x"]
    if out_ref.shape != out_ours.shape:
        raise SystemExit(
            f"输出形状不一致：ref {tuple(out_ref.shape)} vs ours {tuple(out_ours.shape)}，"
            "请检查架构（registers/pos_embed/head）"
        )
    diff = (out_ours - out_ref).abs().max().item()
    if not torch.allclose(out_ours, out_ref, atol=1e-4):
        raise SystemExit(f"前向不一致 max|diff|={diff:.4e} > 1e-4，键名/架构仍有错位，请调整")
    print(f"  前向一致 ✓ (max|diff|={diff:.2e})")

    print(f"[4/5] 打包到 {PRETRAINED}")
    PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": ours.state_dict(), "source": f"{args.hub_repo}:{args.entry}",
         "img_size": args.img_size, "num_register_tokens": n_reg},
        PRETRAINED,
    )
    sha = hashlib.sha256(PRETRAINED.read_bytes()).hexdigest()
    print(f"  sha256: {sha}")

    print("[5/5] 更新 pretrained_manifest.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "name": "franca_vitb14_imagenet21k",
        "description": "Franca（CVPR 2026）ViT-B/14 主干 ImageNet-21K 预训练权重（自监督嵌套 matryoshka 聚类）",
        "source_url": f"https://github.com/{args.hub_repo} (torch.hub:{args.entry})",
        "sha256": sha,
        "local_path": "model/pretrained/franca_vitb14.pth",
        "license": "TODO_VERIFY",  # 实现期核实 Franca 仓库许可后填写
        "usage": "src/train.py 初始化 PatchCore 共享主干（src/patchcore/vit.py），冻结特征提取",
    }
    manifest["entries"] = [e for e in manifest.get("entries", []) if e.get("name") != "franca_vitb14_imagenet21k"]
    manifest["entries"].insert(0, entry)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成。请手动核实：")
    print("  1) Franca 仓库/权重许可证，填入 pretrained_manifest.json 的 license 与 third_party/LICENSES.md")
    print("  2) 删除旧权重 model/pretrained/vit_b_16.pth 与 wide_resnet50_2.pth 缩小 zip")


if __name__ == "__main__":
    main()
