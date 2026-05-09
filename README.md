# Lightweight-Speech-Denoising.axera — 轻量级语音增强部署方案

本工程融合了 [RNNoise](https://github.com/xiph/rnnoise) 的 DSP 框架与 [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) 的模型结构，
在保留较好降噪效果的前提下，面向 Axera NPU 平台进行极致轻量化设计。

**主要特点：**
- **效果优先**：基于 GTCRN 框架训练，在低参数量下仍具备较强降噪能力
- **极致轻量**：最小模型体积不足 100 KB，CMM 占用少于 150 KB
- **算子精简**：tiny_v5 / conv_se 为纯卷积模型，算子种类极少；其中 tiny_v5 可在算子支持有限的 AX525 平台量化部署
- **流程完整**：提供从 ONNX 导出、量化校准到 C 板端推理的端到端部署流程
- **多平台覆盖**：支持 x86、AX620Q、AX630C、AX650 推理及 AX620Q、AX630C、AX650、AX620L、AX525 NPU 量化

## 平台支持

| 模型 | x86 Python | x86 C | AX620Q | AX630C | AX650 | AX620L | AX525 |
|---|---|---|---|---|---|---|---|
| tiny_v5 | ✅ | ✅ | ✅ | ✅ | ✅ | 可量化 | 可量化 ⁽¹⁾ |
| conv_se | ✅ | ✅ | ✅ | ✅ | ✅ | 可量化 | 不支持 |
| GTCRN   | ✅ | ✅ | ✅ | ✅ | ✅ | 可量化 | 不支持 |

> - AX620L / AX525：量化已支持，板端推理待后续更新。
> - ⁽¹⁾ AX525 目前仅支持 tiny_v5 量化，且校准数据格式与配置文件与其他平台不同，后续单独补充。

---

## 目录结构

```
Lightweight-Speech-Denoising.axera/
├── README.md
├── axmodels/                       # 量化后的 axmodel
├── result/                         # 各平台板端推理结果音频
│   ├── ax620q/
│   ├── ax630c/
│   └── ax650/
│
├── c_infer/                        # C 推理工程
│   ├── CMakeLists.txt              # CMake 构建（x86 / ax620q / ax630c / ax650）
│   ├── src/
│   │   ├── ax_ai_se_denoise.c      # x86 ONNX 推理
│   │   ├── ax_ai_se_denoise_ax.c   # AX NPU 推理
│   │   ├── test_cache_denoise.c    # x86 主测试程序
│   │   └── test_se_denoise_ax.c    # 板端主测试程序
│   ├── inc/
│   │   ├── ax_ai_se_denoise.h      # 推理引擎公共接口
│   │   └── ax_base_type.h          # 基础类型定义
│   ├── lib/
│   │   ├── tiny_se_v5_dsp.c/h      # STFT/iSTFT、对数幅度特征提取
│   │   └── rnnoise_src/            # kiss_fft + rnnoise DSP 库
│   ├── models/                     # config.ini 文件；ONNX 和 axmodel 放此处
│   │   └── *_config.ini            # 各平台各模型配置文件
│   ├── toolchain_ax620q.cmake      # AX620Q 交叉编译工具链配置
│   ├── toolchain_ax650.cmake       # AX650/AX630C 交叉编译工具链配置
│   ├── build_x86.sh                # x86 编译
│   ├── build_ax620q.sh             # AX620Q/E 交叉编译
│   ├── build_ax630c.sh             # AX630C 交叉编译
│   ├── build_ax650.sh              # AX650 交叉编译
│   ├── run_x86_all.sh              # x86 批量推理
│   ├── run_ax620q_all.sh           # AX620Q 板端批量推理
│   ├── run_ax630c_all.sh           # AX630C 板端批量推理
│   └── run_ax650_all.sh            # AX650 板端批量推理
│
└── model_convert/                  # 模型导出、Python 推理及量化数据生成（见 model_convert/README.md）
    ├── configs/
    │   └── model_catalog.json          # 模型信息
    ├── checkpoints/                # 预训练权重
    ├── model_src/                  # 模型源码（PyTorch）
    ├── export/                     # ONNX 导出脚本 & 校准数据生成脚本
    │   ├── export_self_models.py
    │   ├── export_gtcrn_ax620e.py
    │   ├── export_gtcrn_ax650.py
    │   └── generate_all.py
    ├── quant/                      # 量化工作区
    │   ├── models/                 # 导出的 ONNX 文件
    │   ├── ax_configs/             # Pulsar2 量化配置
    │   └── calibration_data/       # 量化校准数据
    ├── python/                     # Python 推理脚本
    │   ├── denoise_core.py
    │   ├── infer.py                # ONNX 推理入口
    │   ├── infer_gtcrn_cache_onnx.py
    │   └── requirements.txt
    ├── test_wavs/
    │   └── mix.wav                 # 测试用含噪音频（16kHz）
    └── README.md                   # ONNX 导出 & 量化数据生成详细说明
```

---

## 模型说明

| 模型 | ONNX 文件 | 输入形状 | 输出形状 | 推理模式 |
|---|---|---|---|---|
| **tiny_v5** | `tiny_v5_context.onnx` | `(1,1,34,257)` | `(1,1,34,17)` 频率掩膜 | 滑窗，step=6 |
| **conv_se** | `conv_se_context.onnx` | `(1,1,64,257)` | `(1,1,64,129)` 频率掩膜 | 滑窗，step=6 |
| **GTCRN** | `gtcrn_no_scatter_less_input_optimized.onnx` | `(1,257,1,2)` + 6 cache 张量 | `(1,257,1,2)` + 6 cache 张量 | 帧级 cache |

STFT 参数（三个模型相同）：`n_fft=512, hop_len=256, win_len=512, sample_rate=16000`

---

## 完整流程

### 步骤一：环境准备

```bash
cd Lightweight-Speech-Denoising.axera/model_convert
pip install -r python/requirements.txt
# 主要依赖：torch, onnxruntime, onnx, onnxsim, scipy, soundfile, einops, omegaconf
```
#### pyaxengine

pyaxengine 是 NPU 的 Python API，用于板端 axmodel 推理：

```bash
# 详细安装请参考：https://github.com/AXERA-TECH/pyaxengine
pip install pyaxengine
```

---

### 步骤二～五：ONNX 导出、推理验证、量化数据生成、Pulsar2 量化

> 详见 **[model_convert/README.md](model_convert/README.md)**

快速执行：

```bash
cd Lightweight-Speech-Denoising.axera/model_convert
pip install -r python/requirements.txt

# 导出 ONNX
python3 export/export_self_models.py
python3 export/export_gtcrn_ax620e.py
python3 export/export_gtcrn_ax650.py

# Python 推理验证
python3 python/infer.py --model tiny_v5 --input test_wavs/mix.wav --output out_tiny_v5.wav

# 生成量化校准数据
python3 export/generate_all.py --model all --num_samples 100

# Pulsar2 量化
cd quant
pulsar2 build --config ax_configs/config_tiny_v5_context_620E.json
pulsar2 build --config ax_configs/config_tiny_v5_context_620L.json
pulsar2 build --config ax_configs/config_tiny_v5_context_650.json
pulsar2 build --config ax_configs/config_conv_se_context_620E.json
pulsar2 build --config ax_configs/config_conv_se_context_620L.json
pulsar2 build --config ax_configs/config_conv_se_context_650.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_620E.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_620L.json
pulsar2 build --config ax_configs/config_gtcrn_no_scatter_less_input_optimized_650.json
```

量化完成后将 `*.axmodel` 放入 `axmodels/`（板端 config.ini 中 `model_path` 已指向该目录）。

---

### 步骤六：C 推理

#### x86 编译与运行

```bash
cd Lightweight-Speech-Denoising.axera/c_infer

# 编译（需系统已安装 cmake、gcc；ONNX Runtime 库在 c_infer/third_party/onnxruntime/）
bash build_x86.sh

# 批量推理（三模型）
bash run_x86_all.sh ../test_wavs/mix.wav output/x86_all/
# 输出：
#   output/x86_all/out_tiny_v5_context_c_onnx.wav
#   output/x86_all/out_conv_se_context_c_onnx.wav
#   output/x86_all/out_gtcrn_7input_c_onnx.wav
```

> x86 C 推理使用原始 ONNX，需先将 `model_convert/quant/onnx_models/tiny_v5_context.onnx`、`conv_se_context.onnx`、
> `gtcrn_no_scatter_less_input_optimized.onnx` 拷贝（或符号链接）到 `c_infer/models/`，
> 文件名与 config.ini 中 `model_path` 一致。

#### 交叉编译

编译前先编辑对应 `build_*.sh`，将 `BSP_SDK_DIR` 和 `TOOLCHAIN_DIR` 改为实际路径：

```bash
bash build_ax620q.sh   # AX620Q/E  → build_ax620q/test_se_denoise_ax（ARM32 uclibc）
bash build_ax630c.sh   # AX630C    → build_ax630c/test_se_denoise_ax（ARM64 glibc）
bash build_ax650.sh    # AX650     → build_ax650/test_se_denoise_ax （ARM64 glibc）
```

#### 板端运行

支持 **AX620Q、AX630C、AX650**。AX525 / AX620L 板端推理暂未支持，待后续更新。

将以下文件/目录按原始层级结构上传到板端：

- `c_infer/build_ax650/test_se_denoise_ax`（或对应平台的编译产出）
- `c_infer/models/`（含各平台 config.ini）
- `c_infer/run_ax650_all.sh`（或对应平台的运行脚本）
- `axmodels/`（含量化后的 `.axmodel` 文件）
- `test_wavs/mix.wav`

上传后执行 `run_ax620q_all.sh` / `run_ax630c_all.sh` / `run_ax650_all.sh`。

各平台降噪结果音频已保存至 `result/ax620q/`、`result/ax630c/`、`result/ax650/`。

**AX620Q 执行结果：**

```
# sh run_ax620q_all.sh

[tiny_v5]   Avg infer: 0.736 ms  |  RTF: 0.0332  |  Realtime: 30.1x
[conv_se]   Avg infer: 14.938 ms |  RTF: 0.1970  |  Realtime: 5.1x
[gtcrn]     Avg infer: 3.535 ms  |  RTF: 0.2295  |  Realtime: 4.4x
```

**AX630C 执行结果：**

```
# sh run_ax630c_all.sh

[tiny_v5]   Avg infer: 0.587 ms  |  RTF: 0.0232  |  Realtime: 43.1x
[conv_se]   Avg infer: 7.803 ms  |  RTF: 0.1092  |  Realtime: 9.2x
[gtcrn]     Avg infer: 2.835 ms  |  RTF: 0.1820  |  Realtime: 5.5x
```

**AX650 执行结果：**

```
# sh run_ax650_all.sh

[tiny_v5]   Avg infer: 0.160 ms  |  RTF: 0.0117  |  Realtime: 85.3x
[conv_se]   Avg infer: 1.963 ms  |  RTF: 0.0365  |  Realtime: 27.4x
[gtcrn]     Avg infer: 2.766 ms  |  RTF: 0.1756  |  Realtime: 5.7x
```

---

## 参考

- **RNNoise** — Mozilla 开源 DSP + RNN 降噪框架，本工程复用其 STFT/iSTFT 及 kiss_fft 实现
  https://github.com/xiph/rnnoise

- **GTCRN** — 轻量级 Gated Temporal Convolutional Recurrent Network 语音增强模型
  https://github.com/Xiaobin-Rong/gtcrn
