# 第三方组件与预训练权重许可声明

本仓库/提交包使用以下第三方代码、算法与预训练权重。运行时仅依赖
torch / torchvision / numpy / Pillow（见 `requirements.lock`）。

## 运行依赖（Python 包）

| 包 | 版本 | 许可证 |
|----|------|--------|
| torch | 见 requirements.lock | BSD-3-Clause |
| torchvision | 见 requirements.lock | BSD-3-Clause |
| numpy | 见 requirements.lock | BSD-3-Clause |
| Pillow | 见 requirements.lock | HPND |

## 预训练权重

| 权重 | 来源 | 许可证 | 本地文件 |
|------|------|--------|----------|
| Franca ViT-B/14（ImageNet-21K 自监督） | [valeoai/Franca](https://github.com/valeoai/Franca)（arXiv:2507.14137, CVPR 2026），torch.hub 入口 `franca_vitb14`（use_rasa_head=False，仅主干） | **TODO_VERIFY**：实现期核实仓库与权重许可后填写 | `model/pretrained/franca_vitb14.pth` |

权重哈希与精确来源 URL 见 `pretrained_manifest.json`（由 `scripts/fetch_franca.py` 一次性生成）。

## 算法与实现声明

- **PatchCore**：本文方案参考 anomalib/PatchCore 技术路线（论文 "Towards Total Recall in Industrial Anomaly Detection", Roth et al., CVPR 2022）。`src/patchcore/` 为 torch 原生原创实现，**未复制** anomalib / patchcore 官方源码。
- **Franca 主干**：`src/patchcore/vit.py` 为 DINOv2 风格 ViT-B/14 的 torch 原生实现，仅复用其**预训练权重**（加载前已将键名对齐到本实现命名空间，见 `scripts/fetch_franca.py`），未复制 Franca 仓库源码。
- **DINOv2 架构归属**：ViT-B/14 结构沿用 DINOv2（Meta AI, arXiv:2304.07193）的公开设计；本实现为独立编写。

> 注：若 Franca 权重许可限制再分发，须在提交前停止使用并联系组委确认；此为前提合规项。
