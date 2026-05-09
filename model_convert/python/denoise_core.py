#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample_poly


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "model_catalog.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    onnx: Path
    n_fft: int
    hop_len: int
    win_len: int
    sample_rate: int
    step: int
    input_freq: int
    output_freq: int
    t_model: int | None
    description: str
    raw: dict[str, Any]


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> dict[str, ModelSpec]:
    catalog_path = Path(path)
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    specs: dict[str, ModelSpec] = {}
    for name, item in data.items():
        onnx_path = Path(item["onnx"]).expanduser()
        if not onnx_path.is_absolute():
            onnx_path = (ROOT.parent / onnx_path) if onnx_path.parts and onnx_path.parts[0] == ROOT.name else (ROOT / onnx_path)
        specs[name] = ModelSpec(
            name=name,
            display_name=item["display_name"],
            onnx=onnx_path.resolve(),
            n_fft=int(item.get("n_fft", 512)),
            hop_len=int(item.get("hop_len", 256)),
            win_len=int(item.get("win_len", 512)),
            sample_rate=int(item.get("sample_rate", 16000)),
            step=int(item.get("step", 6)),
            input_freq=int(item.get("input_freq", 257)),
            output_freq=int(item.get("output_freq", 257)),
            t_model=item.get("t_model"),
            description=item.get("description", ""),
            raw=item,
        )
    return specs


def create_session(model_path: str | Path, intra_threads: int = 1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = int(intra_threads)
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def read_wav_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    orig_sr = int(sr)
    if sr != target_sr:
        base = gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, target_sr // base, sr // base).astype(np.float32)
        sr = target_sr
    return wav.astype(np.float32), int(sr), orig_sr


def write_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav.astype(np.float32), sr)


def stft(x: np.ndarray, n_fft: int, hop_len: int, win_len: int) -> np.ndarray:
    window = get_window("hann", win_len, fftbins=True).astype(np.float32)
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n_frames = (len(x) - win_len) // hop_len + 1
    frames = np.stack(
        [x[i * hop_len:i * hop_len + win_len] * window for i in range(n_frames)],
        axis=0,
    )
    return np.fft.rfft(frames, n=n_fft, axis=-1).T.astype(np.complex64)


def istft(spec: np.ndarray, n_fft: int, hop_len: int, win_len: int, n_samples: int) -> np.ndarray:
    window = get_window("hann", win_len, fftbins=True).astype(np.float32)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=-1)[:, :win_len]
    frames *= window

    out_len = (frames.shape[0] - 1) * hop_len + win_len
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i, frame in enumerate(frames):
        start = i * hop_len
        output[start:start + win_len] += frame
        norm[start:start + win_len] += window ** 2

    output /= np.where(norm > 1e-8, norm, 1.0)
    trim = n_fft // 2
    output = output[trim:out_len - trim]
    if len(output) < n_samples:
        output = np.concatenate([output, np.zeros(n_samples - len(output), dtype=np.float32)])
    return output[:n_samples].astype(np.float32)


def extract_log_mag(spec_real: np.ndarray, spec_imag: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mag = np.sqrt(spec_real.T ** 2 + spec_imag.T ** 2 + eps)
    return np.log1p(mag)[np.newaxis].astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def interp_freq(mask: np.ndarray, target_f: int) -> np.ndarray:
    channels, frames, f_in = mask.shape
    if f_in == target_f:
        return mask.astype(np.float32, copy=False)
    x_in = (np.arange(f_in, dtype=np.float32) + 0.5) / f_in
    x_out = (np.arange(target_f, dtype=np.float32) + 0.5) / target_f
    out = np.empty((channels, frames, target_f), dtype=np.float32)
    for c in range(channels):
        for t in range(frames):
            out[c, t] = np.interp(x_out, x_in, mask[c, t])
    return out


def apply_mask_keep_phase(spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    real = spec.real.astype(np.float32)
    imag = spec.imag.astype(np.float32)
    mag = np.sqrt(real.T ** 2 + imag.T ** 2 + 1e-12)
    phase = np.arctan2(imag.T, real.T + 1e-12)
    enh_mag = mag * mask[0]
    enh_real = (enh_mag * np.cos(phase)).T
    enh_imag = (enh_mag * np.sin(phase)).T
    return (enh_real + 1j * enh_imag).astype(np.complex64)


def model_time_dim(sess: ort.InferenceSession) -> int | None:
    dim = sess.get_inputs()[0].shape[2]
    return None if isinstance(dim, str) or dim is None else int(dim)


def infer_full(sess: ort.InferenceSession, spec: np.ndarray) -> np.ndarray:
    f_full, _ = spec.shape
    f_model = sess.get_inputs()[0].shape[3]
    if f_model is not None and not isinstance(f_model, str) and f_full != int(f_model):
        raise ValueError(f"frequency mismatch: audio={f_full}, model={f_model}")
    feat = extract_log_mag(spec.real.astype(np.float32), spec.imag.astype(np.float32))
    logits = sess.run(["output"], {"input": feat[np.newaxis]})[0][0]
    mask = interp_freq(sigmoid(logits), f_full)
    return apply_mask_keep_phase(spec, mask)


def infer_sliding(sess: ort.InferenceSession, spec: np.ndarray, step: int) -> np.ndarray:
    f_full, n_frames = spec.shape
    input_shape = sess.get_inputs()[0].shape
    t_model = int(input_shape[2])
    f_model = int(input_shape[3])
    if f_full != f_model:
        raise ValueError(f"frequency mismatch: audio={f_full}, model={f_model}")
    if step < 1 or step > t_model:
        raise ValueError(f"step must be in [1, {t_model}], got {step}")

    context = t_model - step
    spec_real = spec.real.astype(np.float32)
    spec_imag = spec.imag.astype(np.float32)
    if context > 0:
        real_ctx = np.repeat(spec_real[:, :1], context, axis=1)
        imag_ctx = np.repeat(spec_imag[:, :1], context, axis=1)
    else:
        real_ctx = np.zeros((f_full, 0), dtype=np.float32)
        imag_ctx = np.zeros((f_full, 0), dtype=np.float32)
    real_pad = np.concatenate([real_ctx, spec_real], axis=1)
    imag_pad = np.concatenate([imag_ctx, spec_imag], axis=1)
    enh_real = np.zeros_like(spec_real)
    enh_imag = np.zeros_like(spec_imag)

    n_chunks = (n_frames + step - 1) // step
    for k in range(n_chunks):
        start = k * step
        end = start + t_model
        if end > real_pad.shape[1]:
            pad_len = end - real_pad.shape[1]
            real_chunk = np.concatenate([real_pad[:, start:], np.zeros((f_full, pad_len), np.float32)], axis=1)
            imag_chunk = np.concatenate([imag_pad[:, start:], np.zeros((f_full, pad_len), np.float32)], axis=1)
        else:
            real_chunk = real_pad[:, start:end]
            imag_chunk = imag_pad[:, start:end]

        feat = extract_log_mag(real_chunk, imag_chunk)
        logits = sess.run(["output"], {"input": feat[np.newaxis]})[0][0]
        mask = interp_freq(sigmoid(logits), f_full)[:, context:context + step]

        out_start = k * step
        out_end = min(out_start + step, n_frames)
        valid = out_end - out_start
        if valid <= 0:
            break

        chunk_spec = spec[:, out_start:out_end]
        enh = apply_mask_keep_phase(chunk_spec, mask[:, :valid])
        enh_real[:, out_start:out_end] = enh.real
        enh_imag[:, out_start:out_end] = enh.imag
    return (enh_real + 1j * enh_imag).astype(np.complex64)


def enhance_array(
    wav: np.ndarray,
    sess: ort.InferenceSession,
    spec: ModelSpec,
    mode: str = "auto",
    step: int | None = None,
) -> tuple[np.ndarray, str]:
    spectrum = stft(wav, spec.n_fft, spec.hop_len, spec.win_len)
    fixed_t = model_time_dim(sess)
    use_sliding = mode == "sliding" or (mode == "auto" and fixed_t is not None)
    if mode == "full":
        use_sliding = False
    if use_sliding:
        step_val = int(step or spec.step)
        enhanced_spec = infer_sliding(sess, spectrum, step_val)
        used_mode = f"sliding(step={step_val})"
    else:
        enhanced_spec = infer_full(sess, spectrum)
        used_mode = "full"
    enhanced = istft(enhanced_spec, spec.n_fft, spec.hop_len, spec.win_len, len(wav))
    return enhanced, used_mode


def enhance_file(
    input_path: str | Path,
    output_path: str | Path,
    spec: ModelSpec,
    mode: str = "auto",
    step: int | None = None,
    threads: int = 1,
) -> dict[str, Any]:
    if not spec.onnx.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {spec.onnx}")
    wav, sr, orig_sr = read_wav_mono(input_path, spec.sample_rate)
    sess = create_session(spec.onnx, threads)
    enhanced, used_mode = enhance_array(wav, sess, spec, mode=mode, step=step)
    write_wav(output_path, enhanced, sr)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "model": spec.name,
        "mode": used_mode,
        "sample_rate": sr,
        "original_sample_rate": orig_sr,
        "input_shape": sess.get_inputs()[0].shape,
        "output_shape": sess.get_outputs()[0].shape,
        "duration_sec": len(wav) / float(sr),
    }
