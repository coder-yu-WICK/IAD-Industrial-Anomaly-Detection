# 样例数据（仅供接口联调）

> ⚠️ 这里只放了 **1 个类别（air_conditioner_filter）的 7 张图**，用于本地跑通 train.py / predict.py 接口。
> **完整数据集（30 类，保密）禁止进仓库**，请通过组委会云盘获取，团队内部共享。

## 本地联调命令

```bash
# 训练（3 张正常样本）
python -u src/train.py \
  --data-root samples/Omni-AD-sample \
  --manifest samples/Omni-AD-sample/train_manifest.csv \
  --output-dir work/model \
  --device cpu --seed 2026 --num-workers 0

# 推理（2 张正常 + 2 张缺陷）
python -u src/predict.py \
  --data-root samples/Omni-AD-sample \
  --manifest samples/Omni-AD-sample/eval_manifest.csv \
  --model-dir work/model \
  --output-dir work/predictions \
  --device cpu --num-workers 0
```

预期产物：`work/predictions/predictions.csv` + `work/predictions/maps/*.png`（16-bit 单通道，与原图同尺寸）。
