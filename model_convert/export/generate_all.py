#!/usr/bin/env python3
"""
generate_all.py — 为3个模型生成量化校准数据（AX NPU Pulsar2 格式）

支持的模型:
  tiny_v5  —— Tiny Conv V5 Context       输入: input (1,1,34,257)
  conv_se  —— ConvGTCRN Small V2 Context  输入: input (1,1,64,257)
  gtcrn    —— GTCRN 7-input Less-Input    输入: mix(1,257,1,2) + 6 caches
  all      —— 依次生成以上全部

Context 模型校准策略:
  1. 读取音频，计算 STFT，提取 log1p 幅度特征
  2. 以 step=6 帧滑窗，warmup 若干窗，之后每隔 skip_steps 窗收集一个样本
  3. 每个样本保存为 (1,1,t_model,257) float32 .npy 文件

GTCRN Cache 模型校准策略:
  1. 读取音频，计算 STFT，帧级驱动模型
  2. warmup 若干帧让 cache 预热
  3. 之后每隔 skip_frames 帧收集一个 (mix + 6 caches) 样本

输出目录结构:
  quantization/calibration_data/
  ├── tiny_v5/
  │   ├── input_tiny_v5.zip          # 含 sample_00000.npy … (1,1,34,257)
  ├── conv_se/
  │   ├── input_conv_se.zip          # 含 sample_00000.npy … (1,1,64,257)
  └── gtcrn/
      ├── mix.zip
      ├── en_conv_cache.zip
      ├── de_conv_cache.zip
      ├── en_tra_cache.zip
      ├── de_tra_cache.zip
      ├── inter_cache_0.zip
      └── inter_cache_1.zip

Usage:
  python3 quantization/generate_all.py --model all \\
      --audio_dir test_data/noisy --num_samples 100

  python3 quantization/generate_all.py --model gtcrn \\
      --audio_dir test_data/noisy --num_samples 200
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 模型元信息
# ---------------------------------------------------------------------------

MODELS = {
    "tiny_v5": {
        "onnx": ROOT / "quant" / "onnx_models" / "tiny_v5_context.onnx",
        "backend": "context",
        "t_model": 34,
        "n_freqs": 257,
        "step": 6,
        "input_name": "input",
    },
    "conv_se": {
        "onnx": ROOT / "quant" / "onnx_models" / "conv_se_context.onnx",
        "backend": "context",
        "t_model": 64,
        "n_freqs": 257,
        "step": 6,
        "input_name": "input",
    },
    "gtcrn": {
        # 使用干净的 less-input ONNX 生成标定数据（无 AX650 改动，数值更精确）
        "onnx": ROOT / "quant" / "onnx_models" / "gtcrn_no_scatter_less_input_optimized.onnx",
        "backend": "gtcrn_cache",
        "n_fft": 512,
        "hop_len": 256,
        "win_len": 512,
    },
}

GTCRN_CACHE_NAMES = [
    "en_conv_cache", "de_conv_cache",
    "en_tra_cache",  "de_tra_cache",
    "inter_cache_0", "inter_cache_1",
]
GTCRN_CACHE_SHAPES = [
    (1, 16, 16, 33), (1, 16, 16, 33),
    (1,  3,  1, 16), (1,  3,  1, 16),
    (1,  1, 33, 16), (1,  1, 33, 16),
]


# ---------------------------------------------------------------------------
# 音频 / 特征工具
# ---------------------------------------------------------------------------

def collect_audio(audio_dir: Path, max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        files.extend(audio_dir.glob(ext))
    files = sorted(files)
    if max_files:
        files = files[:max_files]
    print(f">>> 找到 {len(files)} 个音频文件: {audio_dir}")
    return files


def read_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        print(f"  警告: {path.name} 采样率 {sr}，期望 {target_sr}")
    return audio


def compute_stft(audio: np.ndarray, n_fft: int = 512, hop_len: int = 256,
                 win_len: int = 512) -> np.ndarray:
    """返回复数 STFT，形状 (F, T)。

    使用 sqrt-Hann 窗，确保与训练完全一致:
        torch.stft(x, n_fft=512, hop_length=256, win_length=512,
                   window=torch.hann_window(512).pow(0.5), return_complex=True)
    torch.stft 默认 center=True，两侧反射填充 n_fft//2。
    """
    import torch
    audio_tensor = torch.from_numpy(audio)
    window = torch.hann_window(win_len).pow(0.5)
    spec = torch.stft(
        audio_tensor, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
        window=window, return_complex=True,
    )  # (F, T) complex
    return spec.numpy().astype(np.complex64)  # (F, T)


def extract_log_mag(spec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """spec: (F, T) complex → log1p(|spec|), shape (T, F)。"""
    mag = np.abs(spec).T + eps
    return np.log1p(mag).astype(np.float32)  # (T, F)


# ---------------------------------------------------------------------------
# ZIP 打包
# ---------------------------------------------------------------------------

def pack_dir_to_zip(src_dir: Path, zip_path: Path) -> None:
    npy_files = sorted(src_dir.glob("*.npy"))
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in npy_files:
            zf.write(str(f), f.name)
    size_kb = zip_path.stat().st_size / 1024
    print(f"  ✓ {zip_path.name}: {len(npy_files)} 样本 → {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Context 模型（tiny_v5 / conv_se）
# ---------------------------------------------------------------------------

def generate_context(
    model_name: str,
    spec_meta: dict,
    audio_files: list[Path],
    num_samples: int,
    out_dir: Path,
    skip_steps: int = 5,
    warmup_steps: int = 10,
) -> int:
    import onnxruntime as ort

    onnx_path = spec_meta["onnx"]
    t_model   = spec_meta["t_model"]
    n_freqs   = spec_meta["n_freqs"]
    step      = spec_meta["step"]
    inp_name  = spec_meta["input_name"]

    print(f"\n{'='*60}")
    print(f" 模型: {model_name}  (context, t_model={t_model}, step={step})")
    print(f" ONNX: {onnx_path}")
    print(f"{'='*60}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    tmp_dir = out_dir / "_tmp" / model_name / inp_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    pbar = tqdm(total=num_samples, desc=f"{model_name} 标定")

    # 循环复用音频文件直到凑够 num_samples
    file_cycle = 0
    while sample_count < num_samples:
        audio_file = audio_files[file_cycle % len(audio_files)]
        file_cycle += 1
        try:
            audio = read_audio(audio_file)
            spec  = compute_stft(audio)       # (F, T)
            feat  = extract_log_mag(spec)     # (T, F)
            T_total = feat.shape[0]

            # 滑动窗口起始帧列表
            starts = list(range(0, T_total - t_model + 1, step))
            if not starts:
                continue

            # warmup: 跳过前 warmup_steps 个窗口
            warmup_end = min(warmup_steps, len(starts))
            collect_starts = starts[warmup_end:]

            for idx, start in enumerate(collect_starts):
                if sample_count >= num_samples:
                    break
                if idx % skip_steps != 0:
                    continue
                window = feat[start: start + t_model, :]  # (t_model, F)
                inp_tensor = window[np.newaxis, np.newaxis]  # (1, 1, t_model, F)
                sid = f"sample_{sample_count:05d}"
                np.save(str(tmp_dir / f"{sid}.npy"), inp_tensor)
                sample_count += 1
                pbar.update(1)

            # 避免无限循环（文件太短无法生成任何窗口）
            if file_cycle > len(audio_files) * 200:
                print(f"  警告: 循环次数过多，提前退出（已收集 {sample_count} 个样本）")
                break

        except Exception as exc:
            import traceback
            print(f"\n  警告: {audio_file.name}: {exc}")
            traceback.print_exc()

    pbar.close()
    print(f">>> 收集到 {sample_count} / {num_samples} 个样本")

    # 打包
    zip_path = out_dir / f"input_{model_name}.zip"
    pack_dir_to_zip(tmp_dir, zip_path)

    # 清理临时目录
    shutil.rmtree(out_dir / "_tmp" / model_name, ignore_errors=True)
    return sample_count


# ---------------------------------------------------------------------------
# GTCRN Cache 模型
# ---------------------------------------------------------------------------

def generate_gtcrn(
    spec_meta: dict,
    audio_files: list[Path],
    num_samples: int,
    out_dir: Path,
    skip_frames: int = 10,
    warmup_frames: int = 20,
) -> int:
    import onnxruntime as ort

    onnx_path = spec_meta["onnx"]
    n_fft     = spec_meta["n_fft"]
    hop_len   = spec_meta["hop_len"]
    win_len   = spec_meta["win_len"]

    print(f"\n{'='*60}")
    print(f" 模型: gtcrn  (7-input less-input cache)")
    print(f" ONNX: {onnx_path}")
    print(f"{'='*60}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    all_names = ["mix"] + GTCRN_CACHE_NAMES
    tmp_dirs: dict[str, Path] = {}
    for name in all_names:
        d = out_dir / "_tmp" / "gtcrn" / name
        d.mkdir(parents=True, exist_ok=True)
        tmp_dirs[name] = d

    def init_caches() -> dict[str, np.ndarray]:
        return {n: np.zeros(s, dtype=np.float32)
                for n, s in zip(GTCRN_CACHE_NAMES, GTCRN_CACHE_SHAPES)}

    def update_caches(outs: list[np.ndarray]) -> dict[str, np.ndarray]:
        return {n: outs[i + 1] for i, n in enumerate(GTCRN_CACHE_NAMES)}

    sample_count = 0
    pbar = tqdm(total=num_samples, desc="gtcrn 标定")

    for audio_file in audio_files:
        if sample_count >= num_samples:
            break
        try:
            audio = read_audio(audio_file)
            spec  = compute_stft(audio, n_fft, hop_len, win_len)  # (F, T)
            T_total = spec.shape[1]
            caches  = init_caches()

            # warmup
            for fi in range(min(warmup_frames, T_total)):
                frame = spec[:, fi: fi + 1]           # (F, 1)
                mix   = np.stack([frame.real, frame.imag], axis=-1)[np.newaxis]  # (1,F,1,2)
                outs  = sess.run([], {"mix": mix, **caches})
                caches = update_caches(outs)

            fi = min(warmup_frames, T_total)
            step_idx = 0
            while fi < T_total and sample_count < num_samples:
                frame = spec[:, fi: fi + 1]
                mix   = np.stack([frame.real, frame.imag], axis=-1)[np.newaxis]  # (1,F,1,2)

                if step_idx % skip_frames == 0:
                    sid = f"sample_{sample_count:05d}"
                    np.save(str(tmp_dirs["mix"] / f"{sid}.npy"), mix.astype(np.float32))
                    for cn, cv in caches.items():
                        np.save(str(tmp_dirs[cn] / f"{sid}.npy"), cv)
                    sample_count += 1
                    pbar.update(1)

                outs   = sess.run([], {"mix": mix, **caches})
                caches = update_caches(outs)
                fi       += 1
                step_idx += 1

        except Exception as exc:
            import traceback
            print(f"\n  警告: {audio_file.name}: {exc}")
            traceback.print_exc()

    pbar.close()
    print(f">>> 收集到 {sample_count} / {num_samples} 个样本")

    # 打包
    for name in all_names:
        zip_path = out_dir / f"{name}.zip"
        pack_dir_to_zip(tmp_dirs[name], zip_path)

    shutil.rmtree(out_dir / "_tmp" / "gtcrn", ignore_errors=True)
    return sample_count


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    # 所有模型统一使用本工程 test_wavs/
    default_audio = str(ROOT / "test_wavs")
    gtcrn_default_audio = str(ROOT / "test_wavs")

    parser = argparse.ArgumentParser(description="生成量化校准数据（支持 tiny_v5 / conv_se / gtcrn / all）")
    parser.add_argument("--model",       choices=["tiny_v5", "conv_se", "gtcrn", "all"], default="all")
    parser.add_argument("--audio_dir",   default=default_audio)
    parser.add_argument("--gtcrn_audio_dir", default=gtcrn_default_audio,
                        help="gtcrn 专用音频目录，默认本工程 test_wavs/")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir",  default=str(ROOT / "quant" / "calibration_data"))
    parser.add_argument("--max_audio",   type=int, default=None, help="限制音频文件数量")
    parser.add_argument("--skip_steps",  type=int, default=5,  help="context 模型每隔 N 窗收集一次")
    parser.add_argument("--skip_frames", type=int, default=5,  help="gtcrn 模型每隔 N 帧收集一次")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--warmup_frames", type=int, default=20)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.exists():
        audio_dir = ROOT / "c_infer" / "data"
        print(f">>> audio_dir 不存在，回退到: {audio_dir}")

    gtcrn_audio_dir = Path(args.gtcrn_audio_dir).resolve()
    if not gtcrn_audio_dir.exists():
        gtcrn_audio_dir = audio_dir
        print(f">>> gtcrn_audio_dir 不存在，回退到: {audio_dir}")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    out_root = Path(args.output_dir)

    for model_name in models_to_run:
        spec = MODELS[model_name]
        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if spec["backend"] == "context":
            cur_audio_files = collect_audio(audio_dir, args.max_audio)
            if not cur_audio_files:
                print(f"错误: {model_name} 未找到音频文件！")
                continue
            generate_context(
                model_name, spec, cur_audio_files,
                args.num_samples, out_dir,
                skip_steps=args.skip_steps,
                warmup_steps=args.warmup_steps,
            )
        else:  # gtcrn_cache
            cur_audio_files = collect_audio(gtcrn_audio_dir, args.max_audio)
            if not cur_audio_files:
                print(f"错误: gtcrn 未找到音频文件！")
                continue
            generate_gtcrn(
                spec, cur_audio_files,
                args.num_samples, out_dir,
                skip_frames=args.skip_frames,
                warmup_frames=args.warmup_frames,
            )

    # 清理残余临时目录
    for model_name in models_to_run:
        tmp = out_root / model_name / "_tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print("完成！校准数据目录:")
    for model_name in models_to_run:
        d = out_root / model_name
        zips = list(d.glob("*.zip"))
        total_mb = sum(z.stat().st_size for z in zips) / 1024 / 1024
        print(f"  {model_name:10s}: {len(zips)} zip 文件，共 {total_mb:.2f} MB → {d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
