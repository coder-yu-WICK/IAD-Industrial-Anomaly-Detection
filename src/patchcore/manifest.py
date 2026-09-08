# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""模型清单 JSON（Omni-AD 校赛接口规范，schema 保持逐字不变）。"""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(model_dir: Path, categories: list[str], model_mode: str = "hybrid") -> None:
    """写出 model_manifest.json：format_version / model_mode / checkpoint 路径 / 类别清单。"""
    manifest = {
        "format_version": "omniad-school-model-1.0",
        "model_mode": model_mode,
        "checkpoint": "shared.pth",
        "checkpoint_pattern": "checkpoints/{category}.pth",
        "categories": categories,
        "score_range": [0.0, 1.0],
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
