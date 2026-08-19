<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong> | <a href="README_ja.md">日本語</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="MeowID" width="420">
</p>

<h1 align="center">MeowID: 面向猫个体识别的双专家检索系统</h1>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0-11bfae">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-0875c1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-supported-ee4c2c">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-supported-005ced">
  <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-supported-032b4a">
  <a href="https://huggingface.co/RicePasteM/MeowID-Base"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-models-ffd21e"></a>
  <a href="https://modelscope.cn/models/RicePasteM/MeowID-Base"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-models-624aff"></a>
</p>

<div align="center">
  胡张驰<sup>1,2,*,†</sup>,
  尚艺<sup>2,*</sup>,
  杨皓程<sup>4,2,*</sup>,
  胡琪伟<sup>5,*</sup>,
  和 李煜政<sup>3,*</sup>
</div>

<p></p>

<div align="center"><sub>
  <sup>1</sup> 中国科学技术大学 电子工程与信息科学系<br>
  <sup>2</sup> 合肥工业大学 智能软件工程学院<br>
  <sup>3</sup> 中山大学 软件工程学院<br>
  <sup>4</sup> 西北工业大学 计算机学院<br>
  <sup>5</sup> 北京林业大学 生物科学与技术学院<br>
</sub></div>

<br>

<p align="center">
  <sup>*</sup> 同等贡献 &nbsp;&nbsp; <sup>†</sup> 项目负责人
</p>

MeowID 在面部优先检索框架中结合了独立参数化的面部专家和整体专家。路由专用的检索库支持在不重新训练识别模型的情况下，可扩展地注册新身份。

## 亮点

- **路由感知识别** — 自动为每张图像选择面部或整体检索路径。
- **双专家表征** — 保留面部细节，同时将全局外观作为上下文证据。
- **持久化身份注册表** — 为每只猫注册一张或多张参考图像，并将归一化的 512 维嵌入向量存储在磁盘上。
- **多推理后端** — 为 PyTorch、ONNX Runtime 和 TensorRT 提供统一的 Python 和 CLI 接口。
- **猫提取流水线** — 使用 ECSeg-X 实例分割生成干净、填充后的整体猫裁剪图。
- **训练与部署工具** — 包含配置验证、分布式训练、模型导出、验证和基准测试工具。

## 方法流程

<p align="center">
  <img src="assets/fig_main.jpg" alt="MeowID 方法流程" width="100%">
</p>

<p align="center"><em>MeowID 使用面部优先路由、双识别专家、验证引导的整体提示融合以及路由专用检索库。</em></p>

## 安装

克隆仓库，创建隔离的 Python 环境，并以可编辑模式安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

根据你的工作流安装所需的额外依赖：

```bash
# PyTorch 推理，含面部检测和整体猫裁剪
pip install -e ".[ecpose,ecseg]"

# ONNX Runtime GPU 部署
pip install -e ".[onnx,ecpose,ecseg]"

# ONNX Runtime CPU 部署
pip install -e ".[onnx-cpu,ecpose,ecseg]"

# TensorRT 部署
pip install -e ".[tensorrt,ecpose,ecseg]"

# 开发与测试
pip install -e ".[dev]"
```

CUDA、ONNX Runtime 和 TensorRT 必须与所选后端和主机 GPU 兼容。参考部署环境为 Linux x86_64，Python 3.10，CUDA 12.4，PyTorch 2.4.1，TensorRT 10.8，NVIDIA RTX 3090。

## 模型文件

`MeowID` 接受一个包含 `deployment.json` 以及所选后端识别和姿态模型的模型目录：

```text
artifacts/
├── MeowID-Base/
│   ├── deployment.json
│   ├── meowid_base.pth
│   ├── meowid_base.onnx
│   ├── meowid_base.engine
│   ├── ecpose.pth
│   ├── ecpose.onnx
│   ├── ecpose.engine
│   └── SHA256SUMS
├── ECSeg/
│   ├── deployment.json
│   ├── ecseg_x.safetensors
│   └── SHA256SUMS
└── training_init/
    ├── body_expert.pth
    ├── face_expert.pth
    └── SHA256SUMS
```

大型模型文件托管在 Git 之外，并在以下两个仓库中镜像：

- [Hugging Face — RicePasteM/MeowID-Base](https://huggingface.co/RicePasteM/MeowID-Base)
- [ModelScope — RicePasteM/MeowID-Base](https://modelscope.cn/models/RicePasteM/MeowID-Base)

TensorRT 引擎与构建时使用的 TensorRT 版本和 GPU 架构绑定。部署到不同环境时，请从 ONNX 文件重新构建引擎。

## 快速开始

运行内置的五猫 ICW 示例，为每只猫构建一个小型检索库，并检索一个留存的查询图像：

```bash
python examples/quickstart.py --backend torch --device cuda:0
```

该示例使用 `000000.jpg` 作为查询图像，其余五张图像作为每只猫的注册检索库。每次运行将在 `outputs/icw-five-cat-registry` 下重建一个独立的注册表，并报告五个查询的 Top-1 准确率。当相应的模型文件和运行时可用时，可通过 `--backend` 选择 `onnx` 或 `tensorrt`。

通过 Python API 实现相同的工作流：

```python
from pathlib import Path

from cat_recognition import MeowID

sample_root = Path("examples/icw_sample")
model = MeowID(
    "artifacts/MeowID-Base",
    backend="torch",
    registry="outputs/icw-five-cat-registry",
)

model.registry.clear()
for identity_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
    gallery = sorted(identity_dir.glob("00000[1-5].jpg"))
    model.register(identity_dir.name, gallery, save=False)
model.save_registry()

query = sample_root / "00001380/000000.jpg"
prediction = model.search(query, top_k=5)[0]
print(prediction.matches[0].cat_id, prediction.matches[0].score)
```

注册表以 `registry.json` 和 `embeddings.npz` 的形式持久化存储。尽可能为每只猫注册多个视角的图像；具有可用面部的图像会同时贡献给两个路由专用的检索库，而面部不可用的图像则贡献给整体检索库。

### 支持的输入格式

大多数面向图像的 API 接受以下任一格式：

- 文件路径、目录或 glob 模式；
- `PIL.Image.Image`；
- RGB NumPy 数组；
- 列表或其他受支持输入的可迭代对象。

## Python API

### 检测并对齐猫面部

```python
detections = model.detect("images/*.jpg")
aligned_faces = model.align("images/*.jpg")
```

每个检测结果包含选中的面部得分、标签和关键点。对齐使用与训练相同的配置文件，当未检测到可用面部时返回 `None`。

### 提取路由感知嵌入向量

```python
embeddings = model.embed(["images/a.jpg", "images/b.jpg"])

for item in embeddings:
    print(item.source, item.route, item.embedding.shape)
```

`embed()` 及其别名 `extract_embeddings()` 返回归一化的 512 维表征，并附带路由元数据。

### 管理身份

```python
model.register("cat_002", "images/cat_002/*.jpg")
model.remove("cat_002")

model.save_registry("registries/backup")
model.load_registry("registries/backup")
```

在 `register()` 中使用 `replace=True` 来替换身份已有的条目。注册表搜索使用精确内积排序；由于嵌入向量经过 L2 归一化，得分即为余弦相似度。

### 带接受阈值的搜索

```python
results = model.search(
    "images/query.jpg",
    top_k=10,
    threshold=0.45,
)
```

阈值应根据部署场景具体设定。请使用目标摄像头、距离、光照、遮挡和检索库规模下采集的图像进行校准；不要将内置的离线基准测试结果作为实际运行阈值。

## 整体猫裁剪

`CatCropper` 是一个独立的 ECSeg-X 封装器，无需加载识别和姿态模型即可使用分割功能：

```python
from cat_recognition import CatCropper

cropper = CatCropper(
    "artifacts/ECSeg",
    device="cuda:0",
    threshold=0.4,
)

results = cropper.crop(
    "images/group_photo.jpg",
    top_k=3,
    output_size=512,
    padding=0.06,
    mask_background=True,
)

for index, cat in enumerate(results[0].cats):
    cat.crop.save(f"cat_{index}.png")
```

每个猫结果包含其置信度得分、边界框、分割掩码、裁剪框和 PIL 裁剪图像。`MeowID.crop_cats()` 通过延迟加载的裁剪器提供相同的功能。

## 命令行接口

安装本包后可使用 `meowid` 命令。

### 注册与搜索

```bash
meowid register cat_001 images/cat_001/*.jpg \
  --model artifacts/MeowID-Base \
  --backend tensorrt \
  --device cuda:0 \
  --registry registries/demo

meowid search images/query.jpg \
  --model artifacts/MeowID-Base \
  --backend tensorrt \
  --device cuda:0 \
  --registry registries/demo \
  --top-k 16
```

### 提取嵌入向量

```bash
meowid embed images/*.jpg \
  --model artifacts/MeowID-Base \
  --backend torch \
  --device cuda:0
```

### 裁剪猫

```bash
meowid crop images/*.jpg \
  --model artifacts/ECSeg \
  --device cuda:0 \
  --output-dir outputs/crops \
  --top-k 3 \
  --output-size 512
```

所有推理命令都会将结构化的 JSON 元数据输出到标准输出。

## 导出用于部署

同时导出 ECPose 和 MeowID-Base：

```bash
# 导出 ONNX 和 TensorRT 文件
meowid export \
  --format all \
  --repo-root "$PWD" \
  --output-dir artifacts/MeowID-Base \
  --min-batch 1 \
  --opt-batch 4 \
  --max-batch 16
```

使用 `--fp32` 禁用默认的 TensorRT FP16 构建，使用 `--no-onnxslim` 跳过 ONNXSlim 优化。

| 后端 | 设备 | 说明 |
| --- | --- | --- |
| PyTorch | CPU / CUDA | 参考后端，兼容训练运行时 |
| ONNX Runtime | CPU / CUDA | 可移植图运行时；安装对应的可选依赖 |
| TensorRT | CUDA | 参考环境中测得最低延迟；引擎与运行环境绑定 |

## 参考结果

在 ICW 测试集上的离线检索结果：

| 路由 | Top-1 | mAP |
| --- | ---: | ---: |
| 整体专家 | 51.34% | 59.00% |
| 猫面部专家 | 78.80% | 83.32% |
| MeowID-Base 硬路由 | **75.93%** | **80.45%** |

在一台 RTX 3090 上对 2,846 张 ICW 测试图像进行端到端 batch-1 测量，包括图像解码、预处理、ECPose、对齐、嵌入提取和路由，但不包括模型加载：

| 后端 | 平均延迟 | 吞吐量 |
| --- | ---: | ---: |
| PyTorch FP32 | 94.23 ms | 10.61 张/秒 |
| ONNX Runtime CPU | 478.67 ms | 2.09 张/秒 |
| ONNX Runtime CUDA | 79.88 ms | 12.51 张/秒 |
| TensorRT FP16 | **60.00 ms** | **16.66 张/秒** |

以上数据描述的是内置评估环境下的结果，不保证生产环境性能。请在目标硬件和工作负载上重新运行 `tools/benchmark_deployment.py`。

## 训练与复现

默认实验按身份配对整体猫图像和对齐后的面部裁剪图：

```text
data/icw_split/{train,val,test}/<cat_id>/*.jpg
data/icw_catface_aligned_petface_tight_v1/{train,val,test}/<cat_id>/*.jpg
```

训练前验证数据、专家检查点、标签映射和模型构建：

```bash
python tools/validate_training_setup.py \
  --data-root /path/to/icw_split \
  --face-root /path/to/icw_catface_aligned_petface_tight_v1
```

启动默认的四 GPU 实验：

```bash
PYTHON_BIN=/path/to/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash train_meowid_base.sh \
  --cfg-options \
  data.root=/path/to/icw_split \
  data.face_root=/path/to/icw_catface_aligned_petface_tight_v1
```

## 验证

运行单元测试：

```bash
pytest -q
```

验证部署文件集：

```bash
python tools/verify_deployment.py \
  --artifact-dir artifacts/MeowID-Base \
  --backend tensorrt \
  --device cuda:0
```