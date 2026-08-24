# ResNet 基线：wide_resnet50_2 全 30 类结果与问题分析

> 分支：`resnet`。当前分支默认主干为 **wide_resnet50_2**（`layer2`+`layer3` 特征），
> 与之前 ViT 主干（`vit_b_16`）形成 A/B 对照。本文档记录全量结果、与 ViT 的对比、以及遗留问题与改进方向。
> 最后更新：2026-08-24

---

## 1. 方法概述

- **方法**：PatchCore（无监督，只用各类 `train/good` 正常样本）。
- **主干**：`wide_resnet50_2`（ImageNet-1K V1 预训练，冻结），取 `layer2`（28×28）+ `layer3`（14×14）特征对齐拼接 + 3×3 邻域聚合 → 1536 维 patch 特征。
- **记忆库**：每类正常样本特征建 bank，k-center greedy coreset 采样（`coreset_ratio=0.1`）。
- **打分**：patch 特征与所属类 bank 的最近邻 L2 距离 → 热图（上采样 + 高斯平滑 `sigma=4`）→ 压缩到 `[0,1]`。
- **预处理**：256 resize → 224 center crop → ImageNet 归一化（与 ViT 完全一致，仅主干不同）。

与上一版 ViT 的**唯一差异是主干**，其余（coreset / sigma / 打分 / 评测口径）全部相同，便于直接归因。

---

## 2. 全 30 类结果（宏平均口径）

```
类别                          I-AP    I-F1    P-AP    P-F1   P-AUC
air_conditioner_filter    0.9529  0.9206  0.4337  0.4952  0.9672
battery_piece             1.0000  1.0000  0.1522  0.2738  0.9789
battery_tab2              0.9682  0.9474  0.6790  0.6274  0.9760
button1                   0.9936  0.9610  0.2853  0.3913  0.9514
capacitor1                0.9940  0.9885  0.0215  0.0499  0.8468
capacitor3                0.9924  0.9630  0.4193  0.4893  0.9560
ceramic_wafer             0.9429  0.8929  0.0156  0.0530  0.8475
character_string          0.9979  0.9859  0.6197  0.6065  0.9865
chip1                     0.9323  0.8681  0.3310  0.4397  0.7872
cloth3                    0.8850  0.8000  0.1022  0.2242  0.7160
coffee_cup_lid1           0.7815  0.7292  0.6179  0.7118  0.9912
dumplings2                0.8757  0.9196  0.0604  0.1360  0.9120
ice_cream_stick           1.0000  1.0000  0.3765  0.4673  0.9706
infusion_bottle_bottom5   0.7395  0.8000  0.0048  0.0317  0.9030
infusion_pipe_interface2  0.9895  0.9556  0.5117  0.5748  0.9770
iron_lattice              0.8068  0.8070  0.1229  0.2227  0.8577
led                       0.9621  0.9091  0.2103  0.2191  0.9635
lunch_box_lid             1.0000  1.0000  0.3404  0.4155  0.9608
nameplate7                0.9646  0.9667  0.0033  0.0080  0.6059
remote_control            0.9312  0.8602  0.0571  0.1330  0.8696
resistance13              0.8127  0.8667  0.0529  0.1206  0.9418
screw_thread              0.9757  0.9091  0.1459  0.2038  0.9560
silicon_piece1            0.9502  0.8919  0.2989  0.4457  0.9473
solar_panel               0.9694  0.9189  0.2355  0.3606  0.9613
spindle_top               0.7529  0.9091  0.0243  0.0624  0.9211
wafer2                    0.9979  0.9836  0.0621  0.1713  0.7394
webbing                   1.0000  1.0000  0.2705  0.3850  0.9941
work_piece12              0.7974  0.7500  0.0856  0.1792  0.9765
work_piece14              0.8341  0.8200  0.1451  0.2116  0.9777
work_piece7               0.7335  0.6813  0.1034  0.1873  0.9778
------------------------------------------------------------------------
宏平均                     0.9178  0.9002  0.2263  0.2966  0.9139
```

---

## 3. 与 ViT 主干对比（宏平均）

| 指标 | vit_b_16 | wide_resnet50_2 | Δ |
|---|---|---|---|
| I-AP | 0.8686 | **0.9178** | +0.049 |
| I-F1 | 0.8574 | **0.9002** | +0.043 |
| P-AP | 0.1932 | **0.2263** | +0.033 |
| P-F1 | 0.2438 | **0.2966** | +0.053 |
| P-AUC | 0.8621 | **0.9139** | +0.052 |

**结论**：ResNet 在全部 5 项指标上超过 ViT，其中**像素级提升最大**（P-F1 +0.053、P-AUC +0.052），
与「28×28（ResNet layer2）vs 14×14（ViT patch）特征分辨率」的假设一致——像素级定位最吃特征分辨率。

---

## 4. 问题分析：为什么很多类别像素级「效果一般」

### 4.1 核心症状：图像级强、像素级弱（「能检出，但定位不准」）

P-AP 最低的 ~10 个类别，图像级指标几乎都不差，说明模型知道「这张图有缺陷」，却画不准「缺陷在哪」：

| 类别 | I-AP | P-AP | P-AUC | 特征 |
|---|---|---|---|---|
| nameplate7 | 0.965 | 0.003 | 0.606 | 像素级≈随机 |
| infusion_bottle_bottom5 | 0.740 | 0.005 | 0.903 | |
| ceramic_wafer | 0.943 | 0.016 | 0.848 | |
| capacitor1 | 0.994 | 0.022 | 0.847 | 图像级近满分 |
| spindle_top | 0.753 | 0.024 | 0.921 | |
| resistance13 | 0.813 | 0.053 | 0.942 | |
| remote_control | 0.931 | 0.057 | 0.870 | |
| wafer2 | 0.998 | 0.062 | 0.739 | |

### 4.2 三类根因（按优先级）

1. **小尺寸 / 纹理型缺陷定位不到**：`capacitor1`、`ceramic_wafer`、`wafer2`、`resistance13`、`nameplate7` 都是小器件/文字/陶瓷纹理，缺陷本身极小或呈纹理差异。PatchCore 在 28×28 特征图上做最近邻 + 高斯平滑，对这种**亚网格级别**的细微缺陷天然吃亏。
2. **特征分辨率天花板**：28×28（224/8）仍偏粗。224 输入下每个 patch 对应 8×8 原图像素，再 upsample 回原图时小缺陷被抹平。
3. **`sigma=4` 高斯平滑 + 最近邻打分对「局部纹理缺陷」不敏感**：最近邻 L2 距离擅长捕捉「离群的局部大块」，对「全局一致的细微纹理偏移」不敏感（如 `ceramic_wafer`、`wafer2`）。

### 4.3 反向例证（定位好、图像级一般）

`coffee_cup_lid1`（P-AP 0.618 / P-F1 0.712，全表最高）恰恰是大块、边界清晰的缺陷——印证「大缺陷定位好、小/纹理缺陷定位差」的结论。

---

## 5. 改进方向（待办，尚未动手）

> 按预期收益排序，均不改变「只用正常样本」的无监督前提。

1. **提高特征分辨率**（最直接）：
   - ResNet 加入 `layer1`（56×56）做多尺度拼接（bank 变大，需调 `coreset_ratio` 控速）；
   - 换分层 Transformer（Swin-T/S/B，`28×28` 中间层 + Transformer 架构加分）——见 `docs/` 另议。
2. **调 `sigma`**：4.0 对小缺陷偏大，可试 2.0 / 1.0 的像素级消融。
3. **调 `coreset_ratio`**：0.1 可能丢关键 patch，试 0.25 / 0.5 的 P-AP 收益（代价是 bank 变大、推理变慢）。
4. **嵌套表征学习替代 coreset**：队友在做的方向，理论上用可学习采样替代贪心 k-center，值得并行验证。
5. **打分头改进**：最近邻 L2 之外可试「局部 + 全局」双尺度聚合，改善纹理型缺陷定位。

---

## 6. 复现方法

主干已默认 `wide_resnet50_2`，直接按 README §6/§7 跑即可（`--backbone`/`--layers` 可选）：

```bash
# 训练（30 类，默认 wide_resnet50_2 / layer2+layer3）
python -u src/train.py \
  --data-root <数据根> --manifest <train_manifest.csv> \
  --output-dir work/model_resnet --device cuda:0 --seed 2026 --num-workers 4

# 推理（predict 端从 shared.pth 自动识别主干，无需指定）
python -u src/predict.py \
  --data-root <数据根> --manifest <test_manifest.csv> \
  --model-dir work/model_resnet --output-dir work/pred_resnet \
  --device cuda:0 --num-workers 4

# 评测
python src/evaluate.py --predictions-dir work/pred_resnet \
  --data-root <数据根> --manifest <test_manifest.csv>
```

换回 ViT 跑对照：

```bash
python -u src/train.py ... --backbone vit_b_16 --layers encoder.layers.2 encoder.layers.3
```

> 预训练权重：首次联网自动下载并缓存到 `model/pretrained/wide_resnet50_2.pth`（`pretrained_manifest.json` 的 sha256 待本地下载后补填）。
