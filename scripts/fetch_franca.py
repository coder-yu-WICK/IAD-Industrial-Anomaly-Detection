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
    # 依 ref 实际权重形状判定 SwiGLU 变体与隐藏维（消除猜测，vit.py 无硬编码形状）
    w12 = ref_sd["blocks.0.mlp.w12.weight"]
    w3 = ref_sd["blocks.0.mlp.w3.weight"]
    if w12.shape[0] == 2 * w3.shape[1]:
        mlp_hidden, mlp_fused = w3.shape[1], False  # 标准 SwiGLU：w12 2×hidden，w3 hidden→dim
    elif w12.shape[0] == w3.shape[1]:
        mlp_hidden, mlp_fused = w3.shape[1] // 2, True  # xFormers fused：w12/w3 同宽
    else:
        raise SystemExit(f"无法判定 SwiGLU 变体：w12_out={w12.shape[0]} w3_in={w3.shape[1]}（应为 2:1 或 1:1）")
    print(f"  SwiGLU: hidden={mlp_hidden} fused={mlp_fused} (w12_out={w12.shape[0]}, w3_in={w3.shape[1]})")

    print(f"[2/5] 构建自写 DinoViT(img_size={args.img_size}, registers={n_reg})")
    ours = build_franca_vitb14(
        img_size=args.img_size, num_register_tokens=n_reg,
        mlp_hidden=mlp_hidden, mlp_fused=mlp_fused,
    )

    sd = strip_prefix(dict(ref_sd))
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  缺失 {len(missing)} 键: {list(missing)[:8]}")
        print(f"  多余 {len(unexpected)} 键: {list(unexpected)[:8]}")
        raise SystemExit("键名未完全对齐。请对照 ref 键名调整 vit.py（重点：MLP/LayerScale/mask_token）")
    print("  键名完全对齐 ✓ (0 缺失 / 0 多余)")

    print("[3/5] 前向一致性校验（黄金标准）：randn 输入，ours 与 ref 输出 allclose(atol=1e-4)")
    x = torch.randn(1, 3, args.img_size, args.img_size)

    def _hook_cls(model):
        """采集各阶段 CLS token：patch_embed 输出 + 每 block 输出（定位首个分叉层）。"""
        obs: dict[str, torch.Tensor] = {}

        def _mk(name):
            def h(_m, _i, o):
                if o.dim() == 3:
                    obs[name] = o[0, 0].detach()
            return h

        model.patch_embed.register_forward_hook(_mk("embed"))
        for i, blk in enumerate(model.blocks):
            blk.register_forward_hook(_mk(f"block{i}"))
        return obs

    obs_ref = _hook_cls(ref)
    obs_ours = _hook_cls(ours)
    pre_norm: dict[str, torch.Tensor] = {}
    ours.norm.register_forward_hook(lambda _m, inp, _o: pre_norm.__setitem__("x", inp[0].detach()))
    with torch.no_grad():
        out_ref = ref(x)
        out_ours = ours(x)
    if isinstance(out_ref, (tuple, list)):
        out_ref = out_ref[0]
    if isinstance(out_ref, dict):
        out_ref = out_ref["x"]
    # ref 通常返回 CLS 向量 (B,D)；ours 返回全 token (B,L,D) → 取 ours 的 CLS 对齐
    if out_ref.shape == out_ours.shape:
        ours_cls = out_ours
    elif out_ref.dim() == 2 and out_ref.shape == out_ours[:, 0].shape:
        ours_cls = out_ours[:, 0]
    else:
        raise SystemExit(f"输出形状无法对齐：ref {tuple(out_ref.shape)} vs ours {tuple(out_ours.shape)}")
    # 最终 norm 是否参与 CLS 读取各实现不一，分别比对 norm 前后，任一 ≤1e-4 即通过
    d_post = (ours_cls - out_ref).abs().max().item()
    d_pre = (pre_norm["x"][:, 0] - out_ref).abs().max().item()
    d = min(d_post, d_pre)
    if d > 1e-4:
        print(f"  逐层 CLS max|diff|（首个分叉层定位）:")
        for i in range(12):
            name = f"block{i}"
            dr, do = obs_ref[name], obs_ours[name]
            if dr is not None and do is not None:
                print(f"    {name:>8}: {(do - dr).abs().max().item():.3e}")
        raise SystemExit(
            f"前向不一致 min(|post-norm|={d_post:.4e}, |pre-norm|={d_pre:.4e}) > 1e-4；"
            "结合逐层表定位首个分叉层后对照 vit.py 调整（常见：LayerNorm eps / SwiGLU 顺序 / 注意力数值路径）"
        )
    print(f"  前向一致 ✓ (min|diff|={d:.2e})")

    print(f"[4/5] 打包到 {PRETRAINED}")
    PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": ours.state_dict(), "source": f"{args.hub_repo}:{args.entry}",
         "img_size": args.img_size, "num_register_tokens": n_reg,
         "mlp_hidden": mlp_hidden, "mlp_fused": mlp_fused},
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
