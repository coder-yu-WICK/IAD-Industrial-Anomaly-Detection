# Third-Party Licenses

本文件列明提交包所使用或依赖的第三方组件及其许可。评测断网运行，以下组件均随包提供
（预训练权重）或由评测平台预装（Python 依赖），提交代码不进行任何动态安装。

## Python 依赖（由评测平台按 requirements.lock 提供）

| 组件 | 版本 | License | 用途 |
| ---- | ---- | ------- | ---- |
| PyTorch (`torch`) | 见 `requirements.lock` | BSD-3-Clause | 模型构建 / 训练 / 推理 |
| TorchVision (`torchvision`) | 见 `requirements.lock` | BSD-3-Clause | 主干模型 / 图像预处理 |
| NumPy (`numpy`) | 见 `requirements.lock` | BSD-3-Clause | 数值 / 数组操作 |
| Pillow (`pillow`) | 见 `requirements.lock` | HPND | 图像读取 / 16-bit PNG 输出 |

## 预训练权重（随包存放于 `model/pretrained/`，来源与哈希见 `pretrained_manifest.json`）

| 权重 | License | 说明 |
| ---- | ------- | ---- |
| DINOv2 ViT-L/14 (LVD-142M) `dinov2_vitl14.pth` | Apache-2.0 | 唯一使用的默认主干权重，由 `scripts/fetch_dinov2.py` 校验并打包 |

## 方法参考（未复制代码，仅参考算法思想）

- PatchCore 方法参考 anomalib 技术路线（OpenVINO/anomalib，Apache-2.0）。
  本提交包实现为 torch 原生独立实现（`src/patchcore/`），未包含 anomalib 源码。

## 说明

- 评测平台预装环境或随包权重之外的任何组件均不被本提交包引入或触发下载。
- 若正式评测环境对某个预装依赖的版本有差异，以评测平台实际环境为准；
  `requirements.lock` 仅为本机验证版本，见其头部注释。
