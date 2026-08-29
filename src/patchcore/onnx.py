# Copyright (C) 2026
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime 推理封装：加载截断主干的 .onnx，输出逐层特征（numpy）。

predict 端用它替代 PyTorch 主干前向，省去实例化/加载完整模型的成本，
并借助 ONNX Runtime 的算子融合与优化内核加速特征提取。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxBackbone:
    """轻量 ONNX 主干推理器。

    Args:
        path: train.py 导出的 ``shared.onnx`` 路径。
        providers: 执行后端，默认优先 CUDA，回退 CPU。
    """

    def __init__(
        self,
        path: Path | str,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort  # 延迟导入：未安装 onnxruntime 时回退 PyTorch 路径

        if providers is None:
            available_providers = ort.get_available_providers()
            providers = []
            if "TensorrtExecutionProvider" in available_providers:
                trt_options = {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(Path(path).parent),
                }
                providers.append(("TensorrtExecutionProvider", trt_options))
            providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """x: (1, 3, H, W) float32 → list[(1, C, h, w) float32]，按导出层顺序。"""
        return self.sess.run(self.output_names, {self.input_name: x})
