# IAD - Industrial Anomaly Detection

工业图像异常检测与定位（Omni-AD 校赛）——无监督方案提交包。

> 本仓库即提交包内容（`model/` 中为训练产物，数据不入库）。正式评测在组委会 Linux/CUDA 环境**断网**运行，只调用 `src/train.py` 与 `src/predict.py`。

---

## 方案概述

- **方法**：PatchCore（对齐 anomalib 技术路线）+ **级联剪枝多级检索**，无监督，只用各类别 `train/good` 正常样本。
  - 主干 **Franca**（[valeoai/Franca](https://github.com/valeoai/Franca)，CVPR 2026）ViT-B/14，ImageNet-21K 自监督预训练（嵌套 matryoshka 聚类表征），权重随包离线加载，冻结。
  - 取**浅/中/深**三层 Transformer block 特征（`blocks.3/6/9`，518 输入 → 37×37 token 网格），每层独立建 memory bank，coreset 在 concat 特征上选一次索引、同一索引应用到每层（跨层对齐）。
  - **推理**：patch 特征与所属类别 bank 做**级联 top-k 剪枝检索**——浅层全量取最近邻 top10% → 中层子集取 top10% → 深层极小候选集取最近邻距离；逐级缩小候选集以降低计算复杂度，末层距离经高斯平滑热图压缩到 `[0,1]`。
  - matryoshka 前缀维切片（`prefix_dims`）为可选降算开关，默认关闭（全维精确排序）。
- **模型模式**：`hybrid`（共享主干 `shared.pth` + 每类 bank `checkpoints/<category>.pth`）。
- **实现**：`src/patchcore/` 为 torch 原生实现，仅依赖 torch / torchvision / numpy / Pillow。

## 提交包结构

```
TeamName.zip
├── submission.json
├── README.md
├── requirements.lock
├── report.pdf
├── src/
│   ├── train.py            # 统一训练入口（规范 §6）
│   ├── predict.py          # 统一推理入口（规范 §7）
│   └── patchcore/          # 模型实现（torch 原生）
│       ├── vit.py          # DINOv2 风格 ViT-B/14（加载 Franca 主干）
│       ├── cascade.py      # 级联 top-k 剪枝多级检索
│       └── ...
├── configs/
│   └── default.json        # 默认配置（无绝对路径）
├── model/
│   ├── model_manifest.json
│   ├── shared.pth          # 共享主干 state_dict
│   ├── checkpoints/<category>.pth   # 每类多级 memory bank
│   └── pretrained/franca_vitb14.pth  # Franca 权重（断网训练必需）
├── pretrained_manifest.json          # 预训练权重来源与哈希
└── third_party/
    └── LICENSES.md
```

`scripts/`（含 `fetch_franca.py`）为开发用一次性工具，**不进入提交 zip**。

## 依赖安装

```bash
pip install -r requirements.lock
```

仅 4 项：`torch` / `torchvision` / `numpy` / `Pillow`。torch 与 torchvision 由评测平台按 CUDA 环境提供，`requirements.lock` 为代码验证版本。

## 训练（规范 §6）

```bash
python -u src/train.py \
  --data-root <训练数据根目录> \
  --manifest <train_manifest.csv> \
  --output-dir <模型输出目录> \
  --device cuda:0 --seed 2026 --num-workers 4
```

- 严格按 manifest 读取正常训练图像，不访问 test / ground_truth，支持任意类别子集。
- 产物写入 `--output-dir`：`shared.pth` + `checkpoints/<category>.pth` + `model_manifest.json`，不覆盖提交包内原始 `model/`。
- 预训练权重：`model/pretrained/franca_vitb14.pth` 随包存放，断网环境直接加载；缺失时明确报错（禁止联网下载）。该权重由联网开发机运行 `scripts/fetch_franca.py` 一次性生成。

## 推理（规范 §7）

```bash
python -u src/predict.py \
  --data-root <测试数据根目录> \
  --manifest <eval_manifest.csv> \
  --model-dir <模型目录> \
  --output-dir <预测输出目录> \
  --device cuda:0 --num-workers 4
```

- 按 manifest 逐样本处理，不扫描完整数据目录；`--model-dir` 兼容提交包 `model/` 与 train.py 重训目录。
- 产物：
  - `predictions.csv`：`sample_id,image_score`（∈[0,1]）
  - `maps/<sample_id>.png`：单通道 16-bit PNG（0~65535，/65535 得 [0,1] 分数），与原图同尺寸

## manifest 格式（规范 §5）

```csv
sample_id,category,image_path
ev_000001,air_conditioner_filter,eval_images/air_conditioner_filter/ev_000001.png
```

`image_path` 为相对 `--data-root` 的 POSIX 路径，UTF-8 编码。

## 本地联调（开发用，不属于提交接口）

```bash
python -u src/train.py --data-root <公开数据根> --manifest train_manifest.csv \
  --output-dir work/model --device cpu --seed 2026 --num-workers 0
python -u src/predict.py --data-root <公开数据根> --manifest eval_manifest.csv \
  --model-dir work/model --output-dir work/predictions --device cpu --num-workers 0
python src/evaluate.py --predictions-dir work/predictions \
  --data-root <数据根> --manifest eval_manifest.csv
```

`src/evaluate.py` 输出官方指标（Image-level AP / F1-max、Pixel-level AP / F1-max / AUROC，宏平均），供调试，不入提交包。

## 提交前自检（规范 §12）

- [ ] `submission.json` 的 `team_id` 已改为组委会匿名编号
- [ ] `model/` 为真实训练产物，`model/pretrained/franca_vitb14.pth` 为真实权重且 `pretrained_manifest.json` 哈希已填
- [ ] `pretrained_manifest.json` 的 `license` 与 `third_party/LICENSES.md` 已核实 Franca 许可证（非 TODO）
- [ ] `report.pdf` 已补齐
- [ ] zip 根目录直接含 `submission.json`，无嵌套目录
- [ ] 代码 / 报告 / commit 中无学校信息、绝对路径、盘符
- [ ] `predictions.csv` 与 `maps/` 数量完全一致
- [ ] 在含中文和空格的目录中完成过一次完整训练+推理自检
