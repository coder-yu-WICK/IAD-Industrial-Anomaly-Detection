# 第三方代码与许可说明

本提交包（TeamName.zip）除自有代码外，涉及以下第三方组件与算法参考。提交包内仅包含运行必需的文件；完整许可文本可在各来源获取。

## 1. 运行依赖（由 requirements.lock 声明）

| 依赖 | 版本 | 许可 |
|---|---|---|
| torch | 2.12.0 | BSD-3-Clause |
| torchvision | 0.27.0 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause |
| Pillow | 11.3.0 | HPND（MIT-CMU，前身为 PIL 许可） |

以上依赖不在提交包内，由评测环境按 requirements.lock 安装；此处仅记录其许可归属。

## 2. 预训练权重

| 权重 | 来源 | 许可 |
|---|---|---|
| Franca ViT-B/14（ImageNet-21K，自监督 matryoshka 聚类） | valeoai/Franca（CVPR 2026, arXiv:2507.14137），见 `pretrained_manifest.json` | ⚠️ 待核实（Franca 仓库许可，填入 `pretrained_manifest.json` 与本文档） |
| vit_b_16（ImageNet-1K V1，旧基线遗留） | torchvision 模型库 | BSD-3-Clause（TorchVision 模型权重） |

当前方案使用 `model/pretrained/franca_vitb14.pth`，由联网开发机运行 `scripts/fetch_franca.py` 从 Franca 官方 hub 拉取、键名对齐并前向校验后打包，SHA-256 见 `pretrained_manifest.json`。仅用于通用预训练特征提取，符合校赛"允许使用通用预训练权重"的规定。

## 3. 算法参考与再实现

- **PatchCore**：`Towards Total Recall in Industrial Anomaly Detection`（Roth et al., CVPR 2022）。本提交的 `src/patchcore/` 为该方法的 **torch 原生再实现**（未直接复制参考源码），对齐其技术路线：
  - 主干特征（`franca_vitb14` blocks.3/6/9）+ 3×3 邻域聚合
  - k-center greedy coreset 采样（coreset_ratio=0.1）
  - 最近邻 L2 距离 → 上采样 + 高斯平滑 → [0,1] 异常分数
- **Franca**（CVPR 2026, arXiv:2507.14137）：`src/patchcore/vit.py` 依据 Franca 公布的 DINOv2 风格 ViT-B/14 结构**自写实现**（SwiGLU、LayerScale、iBOT mask_token），并在开发期用 `scripts/fetch_franca.py` 与官方权重做逐层前向一致性校验（atol≤1e-4）；本提交包不含 Franca 源码副本。
- **anomalib**（Apache-2.0）：开发阶段作为行为对齐的参考实现；本提交包不含 anomalib 代码。
- 参考仓库：`amazon-science/patchcore-inspection`（Apache-2.0）。

本实现全部代码（`src/` 下各 `.py` 文件）均为本队原创，文件头以 SPDX 标注 `Apache-2.0`；因未复制第三方源码，故提交包不包含任何第三方源码副本。

## 4. 生成方式

- `requirements.lock`：由开发环境 `pip freeze` 裁剪而来，仅保留代码实际 import 的包。
- `pretrained_manifest.json`：SHA-256 由本地 `model/pretrained/franca_vitb14.pth` 实测生成（运行 `scripts/fetch_franca.py` 后自动填写）。
