#!/usr/bin/env python3
"""
Export GTCRN no-scatter less-input ONNX for AX620E and AX620L quantization.

Output (in denoise_solution_v2_context/models/):
  gtcrn_no_scatter_less_input_optimized.onnx

This 7-input merged-cache format is compatible with both AX620E and AX620L Pulsar2
quantization. (The AX650 variant is handled by export_gtcrn_ax650.py.)

7 inputs:
  mix              (1, 257, 1, 2)
  en_conv_cache    (1, 16, 16, 33)
  de_conv_cache    (1, 16, 16, 33)
  en_tra_cache     (1, 3, 1, 16)
  de_tra_cache     (1, 3, 1, 16)
  inter_cache_0    (1, 1, 33, 16)
  inter_cache_1    (1, 1, 33, 16)

Usage:
    cd denoise_solution_v2_context_sim
    python3 quantization/export_gtcrn_ax620e.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).resolve().parent   # quantization/
ROOT      = THIS_DIR.parent                   # denoise_solution_v2_context/
MODELS_DIR = ROOT / "quant" / "onnx_models"

GTCRN_SRC    = ROOT / "model_src" / "gtcrn"
DEFAULT_CKPT = ROOT / "checkpoints" / "gtcrn" / "model_trained_on_dns3.tar"
PASS_CLEAR_GATHER = THIS_DIR / "pass_clear_gather.py"
DEFAULT_OUT  = MODELS_DIR / "gtcrn_no_scatter_less_input_optimized.onnx"

if str(GTCRN_SRC) not in sys.path:
    sys.path.insert(0, str(GTCRN_SRC))

from gtcrn import GTCRN                              # noqa: E402
from gtcrn_stream import (                           # noqa: E402
    ERB, SFE, ConvBlock, StreamGTConvBlock, DPGRNN, Mask, StreamGTCRN,
)
from modules.convert import convert_to_stream        # noqa: E402


# ── StreamGTCRN with merged (less-input) caches ───────────────────────────────
# Cache layout (same as gtcrn_no_scatter_less_input_optimized.onnx):
#   en_conv_cache (1,16,16,33): frames [0:2]=dil1, [2:6]=dil2, [6:16]=dil5
#   de_conv_cache (1,16,16,33): frames [0:10]=dil5, [10:14]=dil2, [14:16]=dil1
#   en_tra_cache  (1,3,1,16):   3 TRA hidden states stacked
#   de_tra_cache  (1,3,1,16)
#   inter_cache_0 (1,1,33,16):  DPGRNN1 inter cache
#   inter_cache_1 (1,1,33,16):  DPGRNN2 inter cache

class _StreamEncoderLessInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False),
        ])

    def forward(self, x, cc0, cc1, cc2, tc0, tc1, tc2):
        en_outs = []
        x = self.en_convs[0](x); en_outs.append(x)
        x = self.en_convs[1](x); en_outs.append(x)
        x, cc0, tc0 = self.en_convs[2](x, cc0, tc0); en_outs.append(x)
        x, cc1, tc1 = self.en_convs[3](x, cc1, tc1); en_outs.append(x)
        x, cc2, tc2 = self.en_convs[4](x, cc2, tc2); en_outs.append(x)
        return x, en_outs, cc0, cc1, cc2, tc0, tc1, tc2


class _StreamDecoderLessInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2,  (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs, cc0, cc1, cc2, tc0, tc1, tc2):
        x, cc0, tc0 = self.de_convs[0](x + en_outs[4], cc0, tc0)
        x, cc1, tc1 = self.de_convs[1](x + en_outs[3], cc1, tc1)
        x, cc2, tc2 = self.de_convs[2](x + en_outs[2], cc2, tc2)
        x = self.de_convs[3](x + en_outs[1])
        x = self.de_convs[4](x + en_outs[0])
        return x, cc0, cc1, cc2, tc0, tc1, tc2


class StreamGTCRNLessInput(nn.Module):
    """7-input merged-cache GTCRN wrapper for AX620E/AX620L ONNX export."""

    def __init__(self) -> None:
        super().__init__()
        self.erb     = ERB(65, 64)
        self.sfe     = SFE(3, 1)
        self.encoder = _StreamEncoderLessInput()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = _StreamDecoderLessInput()
        self.mask    = Mask()

    def forward(
        self,
        spec: torch.Tensor,          # (1, 257, 1, 2)
        en_conv_cache: torch.Tensor, # (1, 16, 16, 33)
        de_conv_cache: torch.Tensor, # (1, 16, 16, 33)
        en_tra_cache: torch.Tensor,  # (1, 3, 1, 16)
        de_tra_cache: torch.Tensor,  # (1, 3, 1, 16)
        inter_cache_0: torch.Tensor, # (1, 1, 33, 16)
        inter_cache_1: torch.Tensor, # (1, 1, 33, 16)
    ):
        # Split merged caches into per-block slices
        en_cc0 = en_conv_cache[:, :, :2,  :]
        en_cc1 = en_conv_cache[:, :, 2:6, :]
        en_cc2 = en_conv_cache[:, :, 6:,  :]
        de_cc0 = de_conv_cache[:, :, 6:,  :]   # dil5
        de_cc1 = de_conv_cache[:, :, 2:6, :]   # dil2
        de_cc2 = de_conv_cache[:, :, :2,  :]   # dil1

        en_tc0 = en_tra_cache[:, 0, :, :].reshape(1, 1, 16)
        en_tc1 = en_tra_cache[:, 1, :, :].reshape(1, 1, 16)
        en_tc2 = en_tra_cache[:, 2, :, :].reshape(1, 1, 16)
        de_tc0 = de_tra_cache[:, 0, :, :].reshape(1, 1, 16)
        de_tc1 = de_tra_cache[:, 1, :, :].reshape(1, 1, 16)
        de_tc2 = de_tra_cache[:, 2, :, :].reshape(1, 1, 16)
        ic0 = inter_cache_0.reshape(1, 33, 16)
        ic1 = inter_cache_1.reshape(1, 33, 16)

        # Feature extraction
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag  = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        feat = self.erb.bm(feat)
        feat = self.sfe(feat)

        feat, en_outs, en_cc0, en_cc1, en_cc2, en_tc0, en_tc1, en_tc2 = (
            self.encoder(feat, en_cc0, en_cc1, en_cc2, en_tc0, en_tc1, en_tc2)
        )
        feat, ic0 = self.dpgrnn1(feat, ic0)
        feat, ic1 = self.dpgrnn2(feat, ic1)
        m_feat, de_cc0, de_cc1, de_cc2, de_tc0, de_tc1, de_tc2 = (
            self.decoder(feat, en_outs, de_cc0, de_cc1, de_cc2, de_tc0, de_tc1, de_tc2)
        )

        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # Re-merge caches
        en_conv_cache_out = torch.cat([en_cc0, en_cc1, en_cc2], dim=2)
        de_conv_cache_out = torch.cat([de_cc2, de_cc1, de_cc0], dim=2)
        en_tra_cache_out = torch.cat([
            en_tc0.reshape(1, 1, 1, 16),
            en_tc1.reshape(1, 1, 1, 16),
            en_tc2.reshape(1, 1, 1, 16),
        ], dim=1)
        de_tra_cache_out = torch.cat([
            de_tc0.reshape(1, 1, 1, 16),
            de_tc1.reshape(1, 1, 1, 16),
            de_tc2.reshape(1, 1, 1, 16),
        ], dim=1)

        return (
            spec_enh,
            en_conv_cache_out,
            de_conv_cache_out,
            en_tra_cache_out,
            de_tra_cache_out,
            ic0.reshape(1, 1, 33, 16),
            ic1.reshape(1, 1, 33, 16),
        )


def _copy_weights(dst: nn.Module, src: StreamGTCRN) -> None:
    """Copy matching keys from src (StreamGTCRN) into dst."""
    dst_sd, src_sd = dst.state_dict(), src.state_dict()
    for k in dst_sd:
        if k in src_sd:
            dst_sd[k] = src_sd[k]
    dst.load_state_dict(dst_sd)


def _apply_clear_gather(src: Path, dst: Path) -> None:
    """Apply Gather→Slice optimisation pass if pass_clear_gather.py is available."""
    if not PASS_CLEAR_GATHER.exists():
        print(f"  [warning] pass_clear_gather.py not found at {PASS_CLEAR_GATHER}, skipping optimisation")
        import shutil
        shutil.copy2(src, dst)
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("pass_clear_gather", str(PASS_CLEAR_GATHER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.clear_gather(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GTCRN no-scatter less-input ONNX for AX620E/AX620L"
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help="Path to model_trained_on_dns3.tar")
    parser.add_argument("--out",  default=str(DEFAULT_OUT),
                        help="Output path for optimized ONNX")
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path  = Path(args.out)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")

    print("Loading GTCRN checkpoint …")
    model = GTCRN().to(device).eval()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)

    export_model = StreamGTCRNLessInput().to(device).eval()
    _copy_weights(export_model, stream_model)

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    dummy_inputs = (
        torch.randn(1, 257, 1, 2, device=device),
        torch.zeros(1, 16, 16, 33, device=device),  # en_conv_cache
        torch.zeros(1, 16, 16, 33, device=device),  # de_conv_cache
        torch.zeros(1,  3,  1, 16, device=device),  # en_tra_cache
        torch.zeros(1,  3,  1, 16, device=device),  # de_tra_cache
        torch.zeros(1,  1, 33, 16, device=device),  # inter_cache_0
        torch.zeros(1,  1, 33, 16, device=device),  # inter_cache_1
    )
    input_names  = ["mix", "en_conv_cache", "de_conv_cache", "en_tra_cache",
                    "de_tra_cache", "inter_cache_0", "inter_cache_1"]
    output_names = ["enh", "en_conv_cache_out", "de_conv_cache_out", "en_tra_cache_out",
                    "de_tra_cache_out", "inter_cache_0_out", "inter_cache_1_out"]

    with torch.inference_mode():
        _ = export_model(*dummy_inputs)

    # ── Raw export ────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out_path.with_name(out_path.stem.replace("_optimized", "") + "_raw.onnx")
    torch.onnx.export(
        export_model,
        dummy_inputs,
        str(raw_out),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    m = onnx.load(str(raw_out))
    onnx.checker.check_model(m)
    print(f"Raw export: {raw_out.name}")

    # ── onnxsim ───────────────────────────────────────────────────────────────
    if not HAS_ONNXSIM:
        raise ImportError("onnxsim is required; pip install onnxsim")
    simple_out = out_path.with_name(out_path.stem.replace("_optimized", "") + "_simple.onnx")
    m_simp, check = onnxsim.simplify(m)
    if not check:
        raise RuntimeError("onnxsim validation failed")
    onnx.save(m_simp, str(simple_out))
    print(f"Simplified:  {simple_out.name}")

    # ── Gather → Slice optimisation ───────────────────────────────────────────
    _apply_clear_gather(simple_out, out_path)
    print(f"Optimized:   {out_path.name}")

    # ── Save calibration .npy files ───────────────────────────────────────────
    for name, tensor in zip(input_names, dummy_inputs):
        npy = out_path.with_name(f"gtcrn_ax620e_{name}.npy")
        np.save(npy, tensor.detach().cpu().numpy().astype(np.float32))
    print("Calibration .npy files saved.")

    # ── Cleanup temp files ────────────────────────────────────────────────────
    raw_out.unlink(missing_ok=True)

    print(f"\nDone. Output: {out_path}")
    print(f"Size: {out_path.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
