#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from denoise_core import DEFAULT_CATALOG, enhance_file, load_catalog, read_wav_mono, write_wav
from infer_gtcrn_cache_onnx import enhance_gtcrn_cache
from infer_gtcrn_cache_onnx import create_session as create_gtcrn_session


def collect_wavs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(path.glob(pattern))


def build_parser() -> argparse.ArgumentParser:
    catalog = load_catalog()
    parser = argparse.ArgumentParser(description="Self-developed denoise ONNX inference")
    parser.add_argument("--model", choices=sorted(catalog), default="tiny_v5")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input", required=True, help="Input wav file or wav directory")
    parser.add_argument("--output", help="Output wav path for single input")
    parser.add_argument("--output-dir", default="output/python")
    parser.add_argument("--mode", choices=["auto", "full", "sliding"], default="auto")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--suffix", default="_se")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = load_catalog(args.catalog)
    spec = catalog[args.model]
    input_path = Path(args.input).resolve()
    wavs = collect_wavs(input_path, args.recursive)
    if not wavs:
        raise FileNotFoundError(f"No wav files found: {input_path}")

    print(f"model: {args.model} ({spec.display_name})")
    print(f"onnx : {spec.onnx}")
    if spec.raw.get("backend") == "gtcrn_cache":
        if not spec.onnx.exists():
            raise FileNotFoundError(f"GTCRN ONNX model does not exist: {spec.onnx}")
        sess = create_gtcrn_session(spec.onnx, args.threads)
        for wav_path in wavs:
            if input_path.is_file():
                out_path = Path(args.output or Path(args.output_dir) / f"{wav_path.stem}_{args.model}{args.suffix}.wav")
            else:
                rel = wav_path.relative_to(input_path)
                out_path = Path(args.output_dir) / args.model / rel
                out_path = out_path.with_name(out_path.stem + args.suffix + out_path.suffix)
            wav, sr, _ = read_wav_mono(wav_path, spec.sample_rate)
            enhanced, elapsed = enhance_gtcrn_cache(wav, sess)
            write_wav(out_path, enhanced, sr)
            print(f"done: {wav_path} -> {out_path} | mode=gtcrn_7input_cache | elapsed={elapsed:.3f}s")
        return

    for wav_path in wavs:
        if input_path.is_file():
            out_path = Path(args.output or Path(args.output_dir) / f"{wav_path.stem}_{args.model}{args.suffix}.wav")
        else:
            rel = wav_path.relative_to(input_path)
            out_path = Path(args.output_dir) / args.model / rel
            out_path = out_path.with_name(out_path.stem + args.suffix + out_path.suffix)
        info = enhance_file(wav_path, out_path, spec, mode=args.mode, step=args.step, threads=args.threads)
        print(
            f"done: {wav_path} -> {out_path} | mode={info['mode']} "
            f"| input_shape={info['input_shape']} | output_shape={info['output_shape']}"
        )


if __name__ == "__main__":
    main()
