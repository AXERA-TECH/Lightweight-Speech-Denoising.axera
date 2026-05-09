#!/usr/bin/env python3
"""
Export tiny_v5 and conv_se models to ONNX for all three AX NPU platforms.

Outputs (in denoise_solution_v2_context/models/):
  tiny_v5_context.onnx        -- for AX620E and AX650 quantization
  tiny_v5_context_ax620l.onnx -- for AX620L quantization (surgery applied)
  conv_se_context.onnx        -- for AX620E and AX650 quantization
  conv_se_context_ax620l.onnx -- for AX620L quantization (surgery applied)

Usage:
    cd denoise_solution_v2_context_sim
    python3 quantization/export_self_models.py
    python3 quantization/export_self_models.py --model tiny_v5
    python3 quantization/export_self_models.py --model conv_se --skip-ax620l
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import onnx
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent            # quantization/
ROOT       = THIS_DIR.parent                            # denoise_solution_v2_context/
MODELS_DIR = ROOT / "quant" / "onnx_models"
QUANT_DIR  = THIS_DIR

CHECKPOINTS = ROOT / "checkpoints"
CONFIGS_DIR = ROOT / "model_src" / "self_configs"
MODELS_SRC  = ROOT / "model_src" / "models"

if str(MODELS_SRC.parent) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC.parent))

from models.model_factory import build_model  # noqa: E402

# ── Per-model specs ────────────────────────────────────────────────────────────
MODEL_SPECS: dict[str, dict] = {
    "tiny_v5": {
        "config":     CONFIGS_DIR / "tiny_v5.yaml",
        "checkpoint": CHECKPOINTS / "tiny_v5" / "model_200.tar",
        "output":     MODELS_DIR / "tiny_v5_context.onnx",
        "output_620l": MODELS_DIR / "tiny_v5_context_ax620l.onnx",
        "surgery_script": QUANT_DIR / "fix_channel_align_ax620l.py",
        "T": 34,
        "F": 257,
        "opset": 18,
    },
    "conv_se": {
        "config":     CONFIGS_DIR / "conv_se.yaml",
        "checkpoint": CHECKPOINTS / "conv_se" / "best_model_122.tar",
        "output":     MODELS_DIR / "conv_se_context.onnx",
        "output_620l": MODELS_DIR / "conv_se_context_ax620l.onnx",
        "surgery_script": QUANT_DIR / "fix_dilation_expand_ax620l.py",
        "T": 64,
        "F": 257,
        "opset": 18,
    },
}


def load_model(spec: dict, device: torch.device) -> tuple[torch.nn.Module, OmegaConf]:
    cfg = OmegaConf.load(spec["config"])
    model = build_model(cfg).to(device)
    try:
        ckpt = torch.load(spec["checkpoint"], map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(spec["checkpoint"], map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    if hasattr(model, "set_export_mode"):
        model.set_export_mode(True)
    return model, cfg


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, output: Path, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed T for AX NPU
    )
    if HAS_ONNXSIM:
        m = onnx.load(str(output))
        m_simp, check = onnxsim.simplify(m)
        if check:
            onnx.save(m_simp, str(output))
            print(f"  [onnxsim] simplified OK")
        else:
            print(f"  [onnxsim] simplification check failed, keeping original")
    else:
        print("  [warning] onnxsim not available, skipping simplification")
    npy_path = output.with_name("input.npy")
    np.save(npy_path, dummy.detach().cpu().numpy().astype(np.float32))
    print(f"  dummy input saved: {npy_path.name}")


def run_surgery(surgery_script: Path) -> None:
    print(f"  running surgery: {surgery_script.name}")
    result = subprocess.run(
        [sys.executable, str(surgery_script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [error] surgery script failed:\n{result.stderr}")
        raise RuntimeError(f"Surgery failed: {surgery_script.name}")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")


def export_model(name: str, spec: dict, device: torch.device, skip_ax620l: bool) -> None:
    print(f"\n{'='*60}")
    print(f"Exporting: {name}")
    print(f"  config:     {spec['config']}")
    print(f"  checkpoint: {spec['checkpoint']}")

    model, cfg = load_model(spec, device)
    dummy = torch.randn(1, 1, spec["T"], spec["F"], device=device)

    with torch.inference_mode():
        out = model(dummy)
    print(f"  model output shape: {out.shape}")

    export_onnx(model, dummy, spec["output"], spec["opset"])
    print(f"  exported: {spec['output'].name}")

    if not skip_ax620l:
        run_surgery(spec["surgery_script"])
        if spec["output_620l"].exists():
            print(f"  ax620l:    {spec['output_620l'].name}")
        else:
            print(f"  [warning] ax620l ONNX not found after surgery: {spec['output_620l']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tiny_v5 and conv_se ONNX models")
    parser.add_argument("--model", choices=["tiny_v5", "conv_se", "all"], default="all",
                        help="Which model(s) to export")
    parser.add_argument("--skip-ax620l", action="store_true",
                        help="Skip ax620L surgery step")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cpu" or not torch.cuda.is_available() else "cpu")

    names = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]
    for name in names:
        export_model(name, MODEL_SPECS[name], device, args.skip_ax620l)

    print(f"\n{'='*60}")
    print("Export complete. Files in models/:")
    for name in names:
        spec = MODEL_SPECS[name]
        for p in [spec["output"], spec["output_620l"]]:
            status = "OK" if p.exists() else "MISSING"
            size = f"{p.stat().st_size // 1024}KB" if p.exists() else "---"
            print(f"  [{status:7s}] {p.name} ({size})")


if __name__ == "__main__":
    main()
