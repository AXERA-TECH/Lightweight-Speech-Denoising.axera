# model_convert — ONNX 导出 & 量化数据生成

> 本目录完成从 PyTorch 权重到 axmodel 的前置准备工作：ONNX 导出、Python 推理验证、量化校准数据生成，以及 Pulsar2 量化命令。

> **说明**：`tiny_v5` 与 `conv_se` 为自研模型，当前暂不开放原始 `.pth` 权重和 `.onnx` 模型；已量化的 `.axmodel` 会随 Releases 提供，可用于板端推理。本工程保留这两个模型的导出、推理和量化命令；当前仅建议执行 GTCRN 的原始模型导出和 x86 ONNX 推理流程。

---

## 目录结构

```
model_convert/
├── README.md
├── checkpoints/            # 预训练 PyTorch 权重
│   └── gtcrn/
│       └── model_trained_on_dns3.tar
├── configs/
│   └── model_catalog.json  # 模型元信息（ONNX 路径、DSP 参数等）
├── model_src/              # PyTorch 模型定义
│   ├── gtcrn/              # GTCRN 模型结构
│   ├── models/             # tiny_v5 / conv_se 等模型结构
│   └── self_configs/       # 自研模型配置
├── export/                 # ONNX 导出脚本 & 量化校准数据生成脚本
│   ├── export_self_models.py
│   ├── export_gtcrn_ax620e.py
│   ├── export_gtcrn_ax650.py
│   ├── generate_all.py
│   ├── verify_all_onnx.py
│   ├── pass_clear_gather.py
│   ├── fix_dilation_expand_ax620l.py
│   └── fix_channel_align_ax620l.py
├── quant/                  # 量化工作区
│   ├── onnx_models/        # 导出的 ONNX 文件
│   ├── ax_configs/         # Pulsar2 量化配置文件
│   └── calibration_data/   # 量化校准数据
├── python/                 # Python 推理脚本
│   ├── infer.py            # ONNX 推理入口
│   ├── denoise_core.py
│   ├── infer_gtcrn_cache_onnx.py
│   └── requirements.txt
└── test_wavs/
    └── mix.wav             # 测试含噪音频（16 kHz）
```

---

## 1. 环境准备

```bash
cd Lightweight-Speech-Denoising.axera/model_convert
pip install -r python/requirements.txt
```

---

## 2. 导出 ONNX


```bash
# cd Lightweight-Speech-Denoising.axera/model_convert

# GTCRN — AX620Q / AX630C / AX620L / AX637
python3 export/export_gtcrn_ax620e.py

# GTCRN — AX650（消除 Explicit Pad 节点）
python3 export/export_gtcrn_ax650.py
```

如具备 `tiny_v5` / `conv_se` 对应原始模型文件，可额外执行：

```bash
# tiny_v5 & conv_se（同时生成 AX620L / AX637 专用 surgery 版本）
python3 export/export_self_models.py
```

**输出文件（`quant/onnx_models/`）：**

| 文件 | 用途 |
|---|---|
| `quant/onnx_models/tiny_v5_context.onnx` | AX620Q / AX630C / AX650 量化 + x86 推理 |
| `quant/onnx_models/tiny_v5_context_ax620l.onnx` | AX620L / AX637 量化专用 |
| `quant/onnx_models/conv_se_context.onnx` | AX620Q / AX630C / AX650 量化 + x86 推理 |
| `quant/onnx_models/conv_se_context_ax620l.onnx` | AX620L / AX637 量化专用 |
| `quant/onnx_models/gtcrn_no_scatter_less_input_optimized.onnx` | AX620Q / AX630C / AX620L / AX637 量化 + x86 推理 |
| `quant/onnx_models/gtcrn_ax650_nopd_fixed.onnx` | AX650 量化专用 |


---

## 3. Python 推理验证

```bash
# cd Lightweight-Speech-Denoising.axera/model_convert

python3 python/infer.py --model gtcrn    --input test_wavs/mix.wav --output out_gtcrn.wav
```

如具备所有 ONNX，可继续推理：

```bash
python3 python/infer.py --model tiny_v5 --input test_wavs/mix.wav --output out_tiny_v5.wav
python3 python/infer.py --model conv_se  --input test_wavs/mix.wav --output out_conv_se.wav
```

---

## 4. 生成量化校准数据

```bash
# cd Lightweight-Speech-Denoising.axera/model_convert
python3 export/generate_all.py --model gtcrn --num_samples 100
```

如具备 `tiny_v5` / `conv_se` 对应原始模型文件，可使用 `python3 export/generate_all.py --model all --num_samples 100` 生成三模型校准数据。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | `tiny_v5` / `conv_se` / `gtcrn` / `all` | `all` |
| `--audio_dir` | 含 wav/flac 的目录（**建议使用含噪音频**） | `test_wavs/` |
| `--num_samples` | 每模型采集样本数 | `100` |

**输出（`quant/calibration_data/`）：**

```
quant/calibration_data/
├── tiny_v5/input_tiny_v5.zip    # N × (1,1,34,257) float32
├── conv_se/input_conv_se.zip    # N × (1,1,64,257) float32
└── gtcrn/mix.zip + 6 cache zips # N × 帧级样本
```

---

## 5. Pulsar2 量化

> 在 Pulsar2 主机上执行，需提前将本工程目录（含 `quant/onnx_models/`、`quant/calibration_data/`和`quant/ax_configs/`）拷贝过去。

```bash
cd Lightweight-Speech-Denoising.axera/model_convert/quant

pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_620E.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_620L.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_637.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_650.json
```

如具备 `tiny_v5` / `conv_se` 对应原始 ONNX，可继续执行：

```bash
cd Lightweight-Speech-Denoising.axera/model_convert/quant

pulsar2 build --config ax_configs/config_tiny_v5_context_620E.json
pulsar2 build --config ax_configs/config_tiny_v5_context_620L.json
pulsar2 build --config ax_configs/config_tiny_v5_context_637.json
pulsar2 build --config ax_configs/config_tiny_v5_context_650.json
pulsar2 build --config ax_configs/config_conv_se_context_620E.json
pulsar2 build --config ax_configs/config_conv_se_context_620L.json
pulsar2 build --config ax_configs/config_conv_se_context_637.json
pulsar2 build --config ax_configs/config_conv_se_context_650.json
```

量化完成后，将 `*.axmodel` 放入 `../../axmodels/`（板端 config.ini 中 `model_path` 已指向该目录）。
