# Swin Transformer 主干（本分支默认）

> 用分层 Transformer（Swin）替代 CNN（wide_resnet50_2）/ 扁平 ViT（vit_b_16），
> 目的是**在保留 Transformer 架构加分的同时，拿到 28×28 的多尺度特征做像素级定位**。
> 最后更新：2026-08-24

---

## 1. 为什么是 Swin

| 主干 | 类型 | 关键特征分辨率 | 像素定位 |
|---|---|---|---|
| vit_b_16 | 扁平 Transformer | 全程 14×14（patch=16） | 粗 |
| wide_resnet50_2 | CNN | layer2 28×28 | 细（但无 Transformer 加分）|
| **swin_t** | 分层 Transformer | **28×28 + 14×14** | 细 + Transformer 加分 |

Swin 的 `features` 是 8 段 Sequential，逐段输出 channel-last 特征图：

```
features[3] 28×28×192  (stage1 blocks)  ← 等价 ResNet layer2
features[5] 14×14×384  (stage2 blocks)  ← 等价 ResNet layer3
```

挂钩 `features.3` + `features.5`（对齐后拼接 → 576 维 patch 特征），与 ResNet layer2+layer3 完全同构。

---

## 2. 代码改动

- `src/patchcore/backbone.py`：新增 `_TruncatedSwin`（截断 `features[0..deepest]`，输出 permute 到 channel-first）；`_truncate` 按 `features`+`head` 特征分派 Swin。
- `src/train.py`：默认 `swin_t` + `("features.3", "features.5")`；`get_pretrained` 改为主干注册表，支持 `swin_t/swin_s/swin_b` + `wide_resnet50_2` + `vit_b_16`。
- `src/predict.py` / `configs/default.json` / `pretrained_manifest.json`：同步 Swin 默认。

---

## 3. 复现

```bash
# 训练（默认 swin_t）
python -u src/train.py \
  --data-root <数据根> --manifest <train_manifest.csv> \
  --output-dir work/model_swin --device cuda:0 --seed 2026 --num-workers 4

# 推理 + 评测（与 ResNet 完全相同）
python -u src/predict.py \
  --data-root <数据根> --manifest <test_manifest.csv> \
  --model-dir work/model_swin --output-dir work/pred_swin \
  --device cuda:0 --num-workers 4
python src/evaluate.py --predictions-dir work/pred_swin \
  --data-root <数据根> --manifest <test_manifest.csv>
```

换更大的 Swin（更多通道，可能更好但更慢）：

```bash
python -u src/train.py ... --backbone swin_b --layers features.3 features.5
```

---

## 4. 预期 & 关注点

- **对比基准**：ResNet 宏平均 `P-AP 0.2263 / P-F1 0.2966 / P-AUC 0.9139`，ViT 宏平均 `P-AP 0.1932 / P-F1 0.2438 / P-AUC 0.8621`（见 [resnet-baseline.md](resnet-baseline.md)）。
- **关注**：P-AP / P-F1 能否追平或超过 ResNet（28×28 分辨率相同，但通道 576 vs ResNet 1536 有差距）。
- **若 swin_t 像素指标偏弱**：试 `swin_b`（768 维通道）或调 `--coreset_ratio` / `sigma`。
- **注意**：Swin 主干 ONNX 导出图较复杂，`train.py` 末尾若导出失败会自动降级 PyTorch（不影响结果）；当前 Colab 上 ONNX 本就不加速，PyTorch 即主线。
