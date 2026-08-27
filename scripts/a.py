#!/usr/bin/env python
"""消融 A：高分辨率 Vit（无嵌套）基线。

仅用高分辨率（518）Franca ViT-B/14，blocks.3/6/9 拼接成单个 concat bank，
无层级联（cascade=False）、无前缀维切片。隔离"主干 + 分辨率"本身贡献。

运行：python scripts/a.py   → 产物 work/ablation_a/
"""
from __future__ import annotations

from _ablation import run

if __name__ == "__main__":
    run("work/ablation_a", cascade=False)
