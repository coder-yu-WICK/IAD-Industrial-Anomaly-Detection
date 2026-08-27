#!/usr/bin/env python
"""消融 B：高分辨率 + 仅层维嵌套。

高分辨率（518）Franca ViT-B/14，blocks.3/6/9 各自独立 bank + 级联 top-k 剪枝
（cascade=True），前缀维关闭（全维精确）。即当前默认方法。

运行：python scripts/b.py   → 产物 work/ablation_b/
"""
from __future__ import annotations

from _ablation import run

if __name__ == "__main__":
    run("work/ablation_b", cascade=True)
