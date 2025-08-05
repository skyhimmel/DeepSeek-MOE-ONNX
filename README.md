# DeepSeek-MOE-ONNX

本项目旨在将 DeepSeek-MoE 模型转换为 ONNX 格式，并提供在不同硬件（CPU/CUDA/TensorRT）上进行推理的示例，同时支持 FP16 和量化推理。

## 项目结构

- `DeepSeek_MOE.py`: DeepSeek-MoE 模型的核心定义。
- `config.py`: 模型配置参数。
- `dataset.py`: 数据集加载和处理。
- `model.pth`: 预训练的 PyTorch 模型权重。
- `model.onnx.svg`: ONNX 模型结构图（SVG 格式）。
- `moe.png`: 模型相关图片。
- `torch_infer.py`: 使用 PyTorch 进行推理的示例。
- `onnx_cpu_infer.py`: 在 CPU 上使用 ONNX Runtime 进行推理的示例。
- `onnx_cuda_infer.py`: 在 CUDA 上使用 ONNX Runtime 进行推理的示例。
- `onnx_fp16_infer.py`: 使用 ONNX Runtime 进行 FP16 推理的示例。
- `onnx_quantize_infer.py`: 使用 ONNX Runtime 进行量化推理的示例。
- `onnx_trt_infer.py`: 在 TensorRT 上使用 ONNX Runtime 进行推理的示例。
- `train.py`: 模型训练脚本。

## 功能特性

- **模型转换**: 将 DeepSeek-MoE PyTorch 模型转换为 ONNX 格式。
- **多平台推理**: 支持在 CPU、CUDA 和 TensorRT 上进行推理。
- **性能优化**: 提供 FP16 和量化推理示例，以提高推理速度和效率。
- **训练与验证**: 包含模型训练和验证脚本。

## 快速开始

1. **安装依赖**: 
   ```bash
   pip install -r requirements.txt # 如果有 requirements.txt 文件
   ```
   （请根据实际情况创建 `requirements.txt` 文件，包含 `torch`, `onnx`, `onnxruntime`, `onnxruntime-gpu`, `tensorrt` 等）

2. **模型转换**: 运行 `train.py` 脚本，它应该会生成 `model.pth`，然后可以将其转换为 ONNX 格式。

3. **推理示例**: 
   - PyTorch 推理: `python torch_infer.py`


