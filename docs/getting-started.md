# 赛前入门指南

> 面向第一次接触工业异常检测的队友。先读「术语速查」建立直觉，再按「技术路线」和「14 天计划」推进。
> 最后更新：2026-08-19

---

## 1. 术语速查（30 秒扫盲）

| 词 | 一句话解释 |
|---|---|
| 无监督异常检测 | 只用正常样本学"正常长什么样"，偏离正常的就是异常 |
| 零样本冷启动 | 新产线一个缺陷样本都没有，算法必须马上能用 |
| 图像级（Image-level） | 这张图有没有缺陷（输出一个分数） |
| 像素级（Pixel-level） | 缺陷在哪个位置（每个像素输出一个分数） |
| F1 | 精确率 × 召回率的调和平均（需设阈值） |
| AP（平均精度） | 不同阈值下 PR 曲线下面积（阈值无关，更稳） |
| AUROC | ROC 曲线下面积，1.0 完美、0.5 瞎猜（阈值无关） |
| Transformer | 基于自注意力的网络架构，用了 +5 分 |
| DINOv3 / CLIP / ImageNet | 允许使用的通用预训练权重 |
| 预训练权重 | 别人在大数据上训好的参数，直接拿来提特征 |
| 单模型多类泛化 | 一个模型处理 30~50 类产品，用了 +2 分 |
| 100ms / 24GB | 推理耗时与显存的两条硬约束 |
| state_dict | PyTorch 保存模型权重的推荐方式 |

**指标直觉**（重要）：异常检测里异常样本远少于正常样本，所以比赛把 **像素级 F1/AP（25 分）放最高**、AUROC（10 分）作参考。**优先优化像素级 F1/AP。**

---

## 2. 技术路线建议

### 2.1 为什么选「冻结预训练 ViT/DINOv3 + 轻量打分头」

无监督异常检测的主流套路：**用预训练模型提特征 → 正常样本的特征聚成一团 → 异常的离团远**。

| 优点 | 对应收益 |
|---|---|
| 主干**冻结**，几乎不训练 | 训练耗时极低（+8 分档） |
| 模型轻量 | 推理 <100ms、显存小（部署 30 分） |
| 基于 Transformer（ViT/DINOv3） | 稳拿 Transformer +5 分 |
| 单模型多类 | 稳拿多类别泛化 +2 分 |
| 14 天能出中上成绩 | 不碰重训练的 SOTA |

### 2.2 推荐方案骨架

1. **特征提取**：DINOv3（或 ImageNet 预训练 ViT）提取多尺度 patch 特征；
2. **记忆库**：训练集正常样本特征构建 memory bank；
3. **打分**：测试样本特征做最近邻（kNN）距离 / 高斯马氏距离 → 得到像素级热图；
4. **图像级分数**：热图聚合（如 max / top-k 平均）→ 一个 float 分数。

> 这是 **PatchCore** 的轻量思路，成熟、易复现、性能基线不错。

### 2.3 备选（进阶）

- 训练一个轻量 Transformer 解码器做"重建/特征蒸馏"，异常即重建误差大；
- 在 memory bank 基础上做 coreset 采样（PatchCore 原版）提升精度与速度。

---

## 3. 环境安装（双机）

统一 **Python 3.12**。先按平台装 torch，再装其余依赖。

### Mac (Apple M4，本地冒烟测试)

```bash
conda create -n iad python=3.12 -y && conda activate iad
pip install torch torchvision
pip install -r requirements.txt
```

### Windows (RTX 5070，主力训练/推理)

```powershell
conda create -n iad python=3.12 -y
conda activate iad
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ RTX 5070 是 Blackwell（sm_120），必须用 **cu128+** 的 PyTorch；**不要装 xformers**（会降级 PyTorch）。
> 5070 只有 **12GB 显存**：推理能跑通 = 稳过 24GB 约束；但训练别做全模型微调。

---

## 4. 14 天作战计划（8/21 – 9/4）

| 阶段 | 时间 | 目标 | 交付物 |
|---|---|---|---|
| P0 跑通 | Day 1–2 | 数据到手，跑通 `inference.py` 接口，能出分 | 可运行 baseline + README |
| P1 锁加分 | Day 3–4 | Transformer 架构 + 单模型多类 + 代码可复现 | 稳定可复现的框架 |
| P2 提精度 | Day 5–9 | 调打分头、多尺度特征、阈值，冲像素 F1/AP | 试跑 1（白嫖评测） |
| P3 迭代 | Day 10–12 | 根据试跑结果优化，写技术报告 | 试跑 2 |
| P4 收尾 | Day 13–14 | 报告定稿 + 最终模型 + 全套代码，最终提交 | 三件套 |

**两条铁律**：
1. **先能跑通，再谈精度**——接口不对、出不了分，一切白搭。
2. **两次试跑是白嫖的官方评测机会**，别没准备好就交。

---

## 5. 团队分工建议（3 人）

| 角色 | 职责 | 可量化贡献证据 |
|---|---|---|
| A 算法/模型 | DINOv3/ViT 特征 + 打分头设计、消融实验 | 模型代码 + 实验结果 |
| B 工程/接口 | `inference.py`、数据加载、耗时/显存优化 | 接口代码 + 性能数据 |
| C 数据/报告 | 数据预处理、指标实现、技术报告撰写 | 指标代码 + 报告章节 |

> 每位成员都要有 **commit 记录 + 对应代码/文档章节**，作为"贡献达标"的证据（晋级按个人贡献核定）。

---

## 6. 常见坑清单

1. **inference.py 接口不对** → 0 分。热图尺寸 = 原图、返回 `float`。
2. **推理 >100ms** → 该项 0 分。全程测耗时。
3. **显存抖动** → 扣分。关 `cudnn.benchmark`、batch=1、预热一次。
4. **学校信息** → 扣 10 分。代码/报告/commit 里都不出现。
5. **数据进 Git** → 泄密。数据目录已 gitignore，绝不 `git add data/`。
6. **用错预训练数据** → 取消资格。只用 ImageNet/DINOv3/CLIP 通用权重。

---

## 7. 参考资料

- Omni-AD 数据集论文（CVPR 2026）
- PatchCore: *Towards Total Recall in Industrial Anomaly Detection*
- DINOv3 权重与论文
- 仓库内：`docs/competition-analysis.md`（完整规则拆解）
