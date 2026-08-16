from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class OnnxRuntimeSession:
    def __init__(self, model_path: str | Path, device: str = "cuda:0") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install ONNX deployment dependencies with `pip install -e .[onnx]`") from exc

        available = set(ort.get_available_providers())
        providers: list[object]
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
            # ORT wheels do not automatically search the CUDA/cuDNN libraries
            # bundled with PyTorch. ORT >=1.21 can preload those pip-installed
            # libraries, which keeps an editable MeowID install self-contained.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            device_id = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        if str(device).startswith("cuda") and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "ONNX Runtime CUDAExecutionProvider could not be initialized; "
                "refusing to silently run the requested CUDA backend on CPU"
            )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self.input_names) - set(inputs)
        if missing:
            raise ValueError(f"Missing ONNX inputs: {sorted(missing)}")
        feed = {name: np.ascontiguousarray(inputs[name]) for name in self.input_names}
        outputs = self.session.run(self.output_names, feed)
        return dict(zip(self.output_names, outputs))


class TensorRTRuntimeSession:
    """TensorRT 10+ runtime using torch CUDA tensors as device buffers."""

    def __init__(self, engine_path: str | Path, device: str = "cuda:0") -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install TensorRT deployment dependencies with `pip install -e .[tensorrt]`"
            ) from exc
        if not str(device).startswith("cuda"):
            raise ValueError("TensorRT requires a CUDA device")
        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self._lock = threading.Lock()

    def _torch_dtype(self, dtype):
        trt = self.trt
        mapping = {
            trt.float32: self.torch.float32,
            trt.float16: self.torch.float16,
            trt.int32: self.torch.int32,
            trt.int64: self.torch.int64,
            trt.int8: self.torch.int8,
            trt.bool: self.torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            mapping[trt.bfloat16] = self.torch.bfloat16
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported TensorRT dtype: {dtype}") from exc

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        missing = set(self.input_names) - set(inputs)
        if missing:
            raise ValueError(f"Missing TensorRT inputs: {sorted(missing)}")
        with self._lock, torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            device_inputs = {}
            for name in self.input_names:
                expected_dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
                tensor = torch.as_tensor(
                    np.ascontiguousarray(inputs[name]),
                    device=self.device,
                    dtype=expected_dtype,
                ).contiguous()
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"TensorRT rejected shape {tuple(tensor.shape)} for {name}")
                device_inputs[name] = tensor

            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"TensorRT could not infer shapes for: {unresolved}")
            device_outputs = {}
            for name in self.output_names:
                shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
                if any(value < 0 for value in shape):
                    raise RuntimeError(f"TensorRT output {name} has unresolved shape {shape}")
                device_outputs[name] = torch.empty(
                    shape,
                    dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)),
                    device=self.device,
                )

            for name, tensor in {**device_inputs, **device_outputs}.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            stream = self.stream
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            stream.synchronize()
            return {
                name: tensor.detach().cpu().numpy()
                for name, tensor in device_outputs.items()
            }
