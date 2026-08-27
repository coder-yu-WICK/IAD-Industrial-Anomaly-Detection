#!/usr/bin/env python
"""消融 C：高分辨率 + 仅维度嵌套。

高分辨率（518）Franca ViT-B/14，blocks.3/6/9 拼接成单个 concat bank
（cascade=False，无层级联），但启用 matryoshka 前缀维切片
（prefix_dims={"concat":768}，保留拼接特征前 768/2304 维）。仅维度嵌套。

运行：python scripts/c.py   → 产物 work/ablation_c/
"""
from __future__ import annotations

from _ablation import run

if __name__ == "__main__":
    run("work/ablation_c", cascade=False, use_prefix_dist=True, prefix_dims={"concat": 768})
