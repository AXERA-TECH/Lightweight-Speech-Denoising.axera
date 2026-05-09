#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from denoise_core import DEFAULT_CATALOG, create_session, load_catalog, read_wav_mono, write_wav


def gtcrn_stft(wav: np.ndarray, n_fft: int = 512, hop_len: int = 256, win_len: int = 512) -> np.ndarray:
    window = torch.hann_window(win_len).pow(0.5)
    spec = torch.stft(torch.from_numpy(wav.astype(np.float32, copy=False)), n_fft, hop_len, win_len, window, return_complex=False)
    return (spec[..., 0].numpy() + 1j * spec[..., 1].numpy()).astype(np.complex64)


def gtcrn_istft(spec: np.ndarray, n_fft: int = 512, hop_len: int = 256, win_len: int = 512, n_samples: int | None = None) -> np.ndarray:
    window = torch.hann_window(win_len).pow(0.5)
    wav = torch.istft(torch.from_numpy(spec), n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window, length=n_samples)
    return wav.numpy().astype(np.float32)


def _init_packed_caches() -> dict[str, np.ndarray]:
    return {
        "conv_cache": np.zeros((2, 1, 16, 16, 33), dtype=np.float32),
        "tra_cache": np.zeros((2, 3, 1, 1, 16), dtype=np.float32),
        "inter_cache": np.zeros((2, 1, 33, 16), dtype=np.float32),
    }


def _init_split_caches() -> dict[str, np.ndarray]:
    return {
        "en_conv_cache_0": np.zeros((1, 16, 2, 33), dtype=np.float32),
        "en_conv_cache_1": np.zeros((1, 16, 4, 33), dtype=np.float32),
        "en_conv_cache_2": np.zeros((1, 16, 10, 33), dtype=np.float32),
        "de_conv_cache_0": np.zeros((1, 16, 10, 33), dtype=np.float32),
        "de_conv_cache_1": np.zeros((1, 16, 4, 33), dtype=np.float32),
        "de_conv_cache_2": np.zeros((1, 16, 2, 33), dtype=np.float32),
        "en_tra_cache_0": np.zeros((1, 1, 16), dtype=np.float32),
        "en_tra_cache_1": np.zeros((1, 1, 16), dtype=np.float32),
        "en_tra_cache_2": np.zeros((1, 1, 16), dtype=np.float32),
        "de_tra_cache_0": np.zeros((1, 1, 16), dtype=np.float32),
        "de_tra_cache_1": np.zeros((1, 1, 16), dtype=np.float32),
        "de_tra_cache_2": np.zeros((1, 1, 16), dtype=np.float32),
        "inter_cache_0": np.zeros((1, 33, 16), dtype=np.float32),
        "inter_cache_1": np.zeros((1, 33, 16), dtype=np.float32),
    }


def _init_less_caches() -> dict[str, np.ndarray]:
    return {
        "en_conv_cache": np.zeros((1, 16, 16, 33), dtype=np.float32),
        "de_conv_cache": np.zeros((1, 16, 16, 33), dtype=np.float32),
        "en_tra_cache": np.zeros((1, 3, 1, 16), dtype=np.float32),
        "de_tra_cache": np.zeros((1, 3, 1, 16), dtype=np.float32),
        "inter_cache_0": np.zeros((1, 1, 33, 16), dtype=np.float32),
        "inter_cache_1": np.zeros((1, 1, 33, 16), dtype=np.float32),
    }


def _update_split_caches(outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "en_conv_cache_0": outputs[1],
        "en_conv_cache_1": outputs[2],
        "en_conv_cache_2": outputs[3],
        "de_conv_cache_0": outputs[4],
        "de_conv_cache_1": outputs[5],
        "de_conv_cache_2": outputs[6],
        "en_tra_cache_0": outputs[7],
        "en_tra_cache_1": outputs[8],
        "en_tra_cache_2": outputs[9],
        "de_tra_cache_0": outputs[10],
        "de_tra_cache_1": outputs[11],
        "de_tra_cache_2": outputs[12],
        "inter_cache_0": outputs[13],
        "inter_cache_1": outputs[14],
    }


def _update_less_caches(outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "en_conv_cache": outputs[1],
        "de_conv_cache": outputs[2],
        "en_tra_cache": outputs[3],
        "de_tra_cache": outputs[4],
        "inter_cache_0": outputs[5],
        "inter_cache_1": outputs[6],
    }


def _enhance_gtcrn_packed(wav: np.ndarray, sess, stream_step: int) -> tuple[np.ndarray, float]:
    spec = gtcrn_stft(wav)
    caches = _init_packed_caches()
    outs: list[np.ndarray] = []
    start_time = time.perf_counter()
    for t in range(0, spec.shape[1], stream_step):
        chunk = spec[:, t:t + stream_step]
        valid = chunk.shape[1]
        if valid < stream_step:
            chunk = np.concatenate([chunk, np.repeat(chunk[:, -1:], stream_step - valid, axis=1)], axis=1)
        mix = np.stack([chunk.real, chunk.imag], axis=-1).astype(np.float32)[np.newaxis]
        outputs = sess.run(None, {"mix": mix, **caches})
        enh = outputs[0]
        caches = {"conv_cache": outputs[1], "tra_cache": outputs[2], "inter_cache": outputs[3]}
        for i in range(valid):
            outs.append(enh[0, :, i, 0] + 1j * enh[0, :, i, 1])
    elapsed = time.perf_counter() - start_time
    return gtcrn_istft(np.stack(outs, axis=1).astype(np.complex64), n_samples=len(wav)), elapsed


def _enhance_gtcrn_split(wav: np.ndarray, sess) -> tuple[np.ndarray, float]:
    spec = gtcrn_stft(wav)
    caches = _init_split_caches()
    outs: list[np.ndarray] = []
    start_time = time.perf_counter()
    for t in range(spec.shape[1]):
        chunk = spec[:, t:t + 1]
        mix = np.stack([chunk.real, chunk.imag], axis=-1).astype(np.float32)[np.newaxis]
        outputs = sess.run(None, {"mix": mix, **caches})
        enh = outputs[0]
        caches = _update_split_caches(outputs)
        outs.append(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1])
    elapsed = time.perf_counter() - start_time
    return gtcrn_istft(np.stack(outs, axis=1).astype(np.complex64), n_samples=len(wav)), elapsed


def _enhance_gtcrn_less(wav: np.ndarray, sess) -> tuple[np.ndarray, float]:
    spec = gtcrn_stft(wav)
    caches = _init_less_caches()
    outs: list[np.ndarray] = []
    start_time = time.perf_counter()
    for t in range(spec.shape[1]):
        chunk = spec[:, t:t + 1]
        mix = np.stack([chunk.real, chunk.imag], axis=-1).astype(np.float32)[np.newaxis]
        outputs = sess.run(None, {"mix": mix, **caches})
        enh = outputs[0]
        caches = _update_less_caches(outputs)
        outs.append(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1])
    elapsed = time.perf_counter() - start_time
    return gtcrn_istft(np.stack(outs, axis=1).astype(np.complex64), n_samples=len(wav)), elapsed


def enhance_gtcrn_cache(wav: np.ndarray, sess) -> tuple[np.ndarray, float]:
    input_names = {inp.name for inp in sess.get_inputs()}
    if "conv_cache" in input_names:
        input_t = sess.get_inputs()[0].shape[2]
        stream_step = 1 if isinstance(input_t, str) or input_t is None else int(input_t)
        return _enhance_gtcrn_packed(wav, sess, stream_step)
    if "en_conv_cache" in input_names:
        return _enhance_gtcrn_less(wav, sess)
    return _enhance_gtcrn_split(wav, sess)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official GTCRN cache ONNX inference")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    spec = load_catalog(args.catalog)["gtcrn"]
    onnx_path = Path(args.onnx).resolve() if args.onnx else spec.onnx
    wav, sr, _ = read_wav_mono(args.input, spec.sample_rate)
    sess = create_session(onnx_path, args.threads)
    enhanced, elapsed = enhance_gtcrn_cache(wav, sess)
    output = Path(args.output or f"denoise_solution_context_gtcrn/output/{Path(args.input).stem}_gtcrn_cache.wav")
    write_wav(output, enhanced, sr)
    duration = len(wav) / float(sr)
    print(f"done: {args.input} -> {output} | onnx={onnx_path} | elapsed={elapsed:.3f}s | RTF={elapsed / max(duration, 1e-6):.3f}")


if __name__ == "__main__":
    main()
