# Swin 主干结果：swin_t 全 30 类结果与主干横向对比

> 分支：`swin`。默认主干 **swin_t**（`features.3` 28×28 + `features.5` 14×14 多尺度特征）。
> 本文档记录 swin_t 全量结果、与 swin_b / ResNet / ViT 的对比，以及遗留问题与改进方向。
> 最后更新：2026-08-26

---

## 1. 方法概述

- **方法**：PatchCore（无监督，只用各类 `train/good` 正常样本）。
- **主干**：`swin_t`（ImageNet-1K V1 预训练，冻结），取 `features.3`（28×28×192）+ `features.5`（14×14×384）特征对齐拼接 + 3×3 邻域聚合 → 576 维 patch 特征。
- **记忆库**：每类正常样本特征建 bank，k-center greedy coreset 采样（`coreset_ratio=0.1`）。
- **打分**：patch 特征与所属类 bank 的最近邻 L2 距离 → 热图（上采样 + 高斯平滑 `sigma=4`）→ 压缩到 `[0,1]`。
- **预处理**：256 resize → 224 center crop → ImageNet 归一化。

与 ResNet 基线（`wide_resnet50_2`）的唯一差异是主干，其余（coreset / sigma / 打分 / 评测口径）全部相同，便于直接归因。

---

## 2. 全 30 类结果（宏平均口径）

```
类别                          I-AP    I-F1    P-AP    P-F1   P-AUC
air_conditioner_filter    0.9606  0.9032  0.3636  0.4213  0.9436
battery_piece             0.8875  0.8889  0.2691  0.4037  0.9790
battery_tab2              0.9796  0.9730  0.5257  0.5249  0.9563
button1                   0.9983  0.9873  0.3251  0.4250  0.9611
capacitor1                0.9702  0.9497  0.0124  0.0224  0.7344
capacitor3                0.9874  0.9398  0.3014  0.4076  0.9404
ceramic_wafer             0.9128  0.8970  0.0069  0.0267  0.8710
character_string          0.9910  0.9722  0.5390  0.5597  0.9780
chip1                     0.9087  0.8621  0.3088  0.3859  0.7952
cloth3                    0.9667  0.9091  0.0673  0.1413  0.7325
coffee_cup_lid1           0.7889  0.7304  0.5174  0.6757  0.9880
dumplings2                0.9206  0.9095  0.1379  0.2225  0.9297
ice_cream_stick           0.9919  0.9873  0.3963  0.4848  0.9543
infusion_bottle_bottom5   0.8863  0.8800  0.0147  0.0629  0.9231
infusion_pipe_interface2  0.9904  0.9565  0.4740  0.5307  0.9710
iron_lattice              0.6342  0.7273  0.0684  0.1723  0.8246
led                       0.8186  0.8000  0.1627  0.1865  0.9460
lunch_box_lid             0.7759  0.8235  0.5799  0.6222  0.9793
nameplate7                0.9830  0.9667  0.0028  0.0070  0.5634
remote_control            0.8833  0.8627  0.0549  0.1697  0.8301
resistance13              0.8286  0.8125  0.0364  0.1039  0.8993
screw_thread              0.9330  0.9153  0.0982  0.1461  0.9284
silicon_piece1            0.9100  0.8434  0.3417  0.4733  0.9231
solar_panel               0.9542  0.8718  0.4577  0.5127  0.9539
spindle_top               0.8296  0.8696  0.0227  0.0746  0.9096
wafer2                    0.9864  0.9835  0.0463  0.1463  0.7941
webbing                   0.8333  0.8000  0.1567  0.3195  0.9856
work_piece12              0.7988  0.7500  0.1277  0.2474  0.9751
work_piece14              0.6568  0.6829  0.1259  0.2231  0.8553
work_piece7               0.4901  0.5185  0.0632  0.1717  0.9467
------------------------------------------------------------------------
宏平均                     0.8819  0.8658  0.2202  0.2957  0.8991
```

---

## 3. 主干横向对比（宏平均）

| 主干 | bank 维 | I-AP | I-F1 | P-AP | P-F1 | P-AUC |
|---|---|---|---|---|---|---|
| vit_b_16（14×14）| 768 | 0.8686 | 0.8574 | 0.1932 | 0.2438 | 0.8621 |
| **swin_t（28×28+14×14）** | 576 | **0.8819** | **0.8658** | **0.2202** | **0.2957** | **0.8991** |
| wide_resnet50_2（28×28+14×14）| 1536 | 0.9178 | 0.9002 | 0.2263 | 0.2966 | 0.9139 |
| swin_b（28×28+14×14）| 768 | 0.8461 | 0.8461 | 0.1559 | 0.2253 | 0.8720 |

**结论**：

1. **swin_t 全面压制 vit_b_16**：P-AP +0.027 / P-F1 +0.052 / P-AUC +0.037，印证「像素级定位吃特征分辨率」——扁平 ViT 全程 14×14 是硬伤，分层 Swin 拿到 28×28 补回。
2. **swin_t 对 ResNet 像素打平、图像微跌**：P-AP −0.006 / P-F1 −0.001 / P-AUC −0.015 基本持平；图像级 I-AP −0.036 / I-F1 −0.034（Swin-T 通道 576 vs ResNet 1536 偏窄所致）。**换取 Transformer 架构加分（+5），净赚。**
3. **swin_b 反而退步（反向例证）**：P-AP −0.064 / P-F1 −0.070，图像级也跌。原因见下节。

---

## 4. 关键结论：swin_t 是甜点，swin_b 反而退步

换更大的 swin_b（768 维通道）**全盘倒退**，不是 bug，是 PatchCore 的「主干越大 ≠ 越准」规律：

1. **特征越抽象**：大主干容量高，早期层为了解 ImageNet 被训得更「语义化、抗形变」，反而抹掉微小缺陷依赖的低层纹理细节。异常检测要「对局部纹理敏感」，不是「对类别判别强」。
2. **维度诅咒**：bank 从 576 → 768 维，coreset 点数不变（`coreset_ratio=0.1`），高维下 kNN 距离判别力下降。
3. 行业里 AD 任务普遍「中型 CNN/Transformer 吊打大型主干」——`wide_resnet50_2` 与 `swin_t` 恰好都落在甜点区。

**决策**：锁定 **swin_t**，不再往大主干试（swin_s 也不必）。提升像素指标的正确方向是**提高特征分辨率**（见 §6），不是加通道。

---

## 5. 问题分析：细纹理 / 微小缺陷类仍是短板

swin_t 的 P-AP 最低 ~10 类与 ResNet 完全同一批，图像级大多不差，说明「能检出，但定位不准」：

| 类别 | I-AP | P-AP | P-AUC | 特征 |
|---|---|---|---|---|
| nameplate7 | 0.983 | 0.003 | 0.563 | 像素≈随机，AUC 也崩 |
| ceramic_wafer | 0.913 | 0.007 | 0.871 | 陶瓷纹理极细微 |
| capacitor1 | 0.970 | 0.012 | 0.734 | 图像级高、AUC 低 |
| infusion_bottle_bottom5 | 0.886 | 0.015 | 0.923 | 定位漂 |
| spindle_top | 0.830 | 0.023 | 0.910 | 纹理细 |
| resistance13 | 0.829 | 0.036 | 0.899 | — |
| wafer2 | 0.986 | 0.046 | 0.794 | — |
| remote_control | 0.883 | 0.055 | 0.830 | — |
| cloth3 | 0.967 | 0.067 | 0.733 | 纹理 + AUC 低 |
| iron_lattice | 0.634 | 0.068 | 0.825 | 图像级也弱 |
| work_piece7 | 0.490 | 0.063 | 0.947 | 图像级最弱 |

根因：`features.3` 28×28（224/8）仍是 8×8 原图像素粒度，小缺陷在 upsample 回原图时被抹平；`sigma=4` 高斯平滑进一步钝化细纹理。反向例证 `coffee_cup_lid1`（P-AP 0.517 / P-F1 0.676 全表最高）恰恰是大块、边界清晰的缺陷。

---

## 6. 下一步（按预期收益排序，均不改变无监督前提）

1. **加 `features.1`（56×56×96）做三尺度拼接**（最直接，冲像素 25 分）：
   - `LAYERS = ("features.1", "features.3", "features.5")`，bank 96+192+384=672 维、分辨率 56×56。
   - `aggregate()` 已支持任意多层（对齐+池化+拼接），改动仅 3 处各一行。
   - 代价：patch 数 784 → 3136 / 图，推理慢约 4×；`coreset_ratio` 可下调控速。
2. **`image_score` 从 `max` 改 top-k 平均**（冲图像 15 分）：对「单点误报」更稳，通常 I-AP / I-F1 有收益。
3. **调 `sigma`**：4.0 对细纹理偏大，试 2.0 / 1.0 的像素级消融。
4. **调 `coreset_ratio`**：0.1 可能丢关键 patch，试 0.25 / 0.5。

---

## 7. 复现

```bash
# 训练（默认 swin_t）
python -u src/train.py \
  --data-root <数据根> --manifest <train_manifest.csv> \
  --output-dir work/model_swin --device cuda:0 --seed 2026 --num-workers 4

# 推理 + 评测（predict 端从 shared.pth 自动识别主干，无需指定）
python -u src/predict.py \
  --data-root <数据根> --manifest <test_manifest.csv> \
  --model-dir work/model_swin --output-dir work/pred_swin \
  --device cuda:0 --num-workers 4
python src/evaluate.py --predictions-dir work/pred_swin \
  --data-root <数据根> --manifest <test_manifest.csv>
```

换 swin_b 跑对照（结果见 §3，已确认退步）：

```bash
python -u src/train.py ... --backbone swin_b --layers features.3 features.5
```

> Colab 一键运行见 [`notebooks/Omni-AD-30_Colab运行_swin.ipynb`](../notebooks/Omni-AD-30_Colab运行_swin.ipynb)。
> 预训练权重首次联网自动下载并缓存到 `model/pretrained/swin_t.pth`（`pretrained_manifest.json` 的 sha256 待本地下载后补填）。
