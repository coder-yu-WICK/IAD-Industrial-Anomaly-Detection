# IAD - Industrial Anomaly Detection

工业图像异常检测与定位 —— 基于无监督范式的算法实现。

> 竞赛环境：训练与推理统一在组委会提供的 AI 训练平台运行，最终提交 Docker 镜像。
> 本仓库仅同步**代码与配置文件**；数据、权重、checkpoint 一律不入库（见 `.gitignore`）。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/competition-analysis.md](docs/competition-analysis.md) | 赛题分析与规则拆解（评分、约束、接口、晋级规则） |
| [docs/getting-started.md](docs/getting-started.md) | 赛前入门指南（术语、技术路线、14 天计划、分工） |

---

## 目录结构

```
IAD-Industrial-Anomaly-Detection/
├── README.md            # 本文件
├── requirements.txt     # 依赖清单（torch 除外，按平台单独装）
├── environment.yml      # conda 环境（Python 3.12）
├── .gitignore           # 排除数据/权重/日志
├── src/                 # 核心代码（模型、训练、推理、指标）
├── configs/             # 实验配置（YAML）
└── scripts/             # 启动/评测脚本
```

## 环境安装

两台机器统一 **Python 3.12**。先按平台装 torch，再装其余依赖。

### 1) 装 PyTorch（按机器选一条）

```bash
# Mac (Apple M4，无 CUDA，仅本地冒烟测试)
pip install torch torchvision

# Windows (RTX 5070, Blackwell, CUDA ≥12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ RTX 5070 是 Blackwell（sm_120），必须用 **cu128 及以上**的 PyTorch；不要装 xformers（会强制降级 PyTorch）。

### 2) 装其余依赖

```bash
pip install -r requirements.txt
```

### 3) 验证 GPU（Windows 上）

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 数据放置

组委会提供的 Omni-AD 子集（正常样本训练集 + 开发测试集）放到本地 `data/` 目录，**该目录已被 gitignore，严禁提交**。

```
data/
├── train/          # 正常样本（无监督训练）
└── test/           # 开发测试集（含标注，本地评测用）
```

## 快速开始

### 统一接口（校赛规范）

训练（只读取 manifest 中的正常样本）：

```bash
python -u src/train.py \
  --data-root <训练数据根目录> \
  --manifest <train_manifest.csv> \
  --output-dir <模型输出目录> \
  --device cuda:0 --seed 2026 --num-workers 4
```

推理（输出 predictions.csv + maps/）：

```bash
python -u src/predict.py \
  --data-root <测试数据根目录> \
  --manifest <eval_manifest.csv> \
  --model-dir <模型目录> \
  --output-dir <预测输出目录> \
  --device cuda:0 --num-workers 4
```

产物格式：

- `predictions.csv`：`sample_id,image_score`（∈[0,1]）
- `maps/<sample_id>.png`：单通道 16-bit PNG（0~65535），与原图同尺寸

模型方案：hybrid（共享主干 `shared.pth` + 每类 memory bank `checkpoints/<category>.pth`），对齐 anomalib PatchCore 技术路线（wide_resnet50_2 + layer2/3 + coreset）。

### manifest 格式

```csv
sample_id,category,image_path
ev_000001,air_conditioner_filter,eval_images/air_conditioner_filter/ev_000001.png
```

`image_path` 为相对 `--data-root` 的 POSIX 路径，UTF-8 编码。

## 注意事项

- **代码 / 报告 / commit 信息中不要出现学校信息。**
- 数据与预训练权重属于保密/受限资源，仅用于本赛题，不外传。
- 方法采用无监督范式，仅允许使用通用预训练权重（ImageNet / DINOv3 / CLIP 等）。
