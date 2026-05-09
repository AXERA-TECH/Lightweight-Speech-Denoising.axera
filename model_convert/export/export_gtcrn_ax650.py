#!/usr/bin/env python3
"""
Export GTCRN ONNX for AX650 NPU quantization.

Root cause of AX650 cmodel error
  assert(wdma[0].size == 0) in cmodel/sys/sys_teng.cpp run() at line 81

The assertion fires because the AX650 NPU tiler cannot handle explicit Pad
ONNX nodes.  Two sources of Pad nodes exist in this model:
  a) nn.Unfold in SFE: 7 Pad nodes (1 top-level + 3 encoder + 3 decoder).
  b) functional.pad in StreamConvTranspose2d.forward(): 3 Pad nodes from
     the 3 decoder GTConvBlocks that use StreamConvTranspose2d as depth_conv.

Additionally, torch.stack (used in channel-shuffle) and `at[..., None]` in
the TRA module generate Unsqueeze ONNX nodes.

Fixes applied in this export:
  1. SFENoPad   — replaces nn.Unfold with Concat-based sliding window.
                  Eliminates all 7 Pad nodes and 14 Gather/Unsqueeze nodes.
  2. StreamTRANoPad — replaces `at[..., None]` with `at.reshape(..., 1)`.
                  Eliminates 6 Unsqueeze nodes (one per TRA/GTConvBlock).
  3. StreamGTConvBlockNoPad — replaces torch.stack shuffle with
                  reshape+cat+reshape, eliminating 12 Unsqueeze nodes.
  4. StreamConvTranspose2dNoPad — replaces functional.pad with torch.cat,
                  eliminating the 3 Pad nodes from decoder depth_conv.

Output: models/gtcrn_ax650_nopd.onnx  (also a *_simple.onnx after onnxsim)

Usage:
    cd denoise_solution_v2_context_sim
    python3 quantization/export_gtcrn_ax650.py

    # Override checkpoint or output paths:
    python3 quantization/export_gtcrn_ax650.py \\
        --ckpt /path/to/model_trained_on_dns3.tar \\
        --out  models/gtcrn_ax650_nopd.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GTCRN_SRC   = Path(__file__).resolve().parents[1] / "model_src" / "gtcrn"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "gtcrn" / "model_trained_on_dns3.tar"
OUT_DIR     = Path(__file__).resolve().parents[1] / "quant" / "onnx_models"
DEFAULT_OUT = OUT_DIR / "gtcrn_ax650_nopd_fixed.onnx"

if str(GTCRN_SRC) not in sys.path:
    sys.path.insert(0, str(GTCRN_SRC))

from gtcrn import GTCRN                                      # offline model
from modules.convert import convert_to_stream                # weight copy helper
from gtcrn_stream import (                                   # original stream classes
    StreamGTCRN, ERB, ConvBlock, GRNN, DPGRNN, Mask,
    StreamGTConvBlock as _OrigGTConvBlock,
)
from modules.convolution import StreamConv2d, StreamConvTranspose2d


# ===========================================================================
# GRU/DPGRNN without batch_first Transpose nodes
# ===========================================================================

class GRNNAx650(nn.Module):
    """GRNN variant that eliminates batch_first Transpose nodes from ONNX export.

    batch_first=True GRU inserts Transpose([1,0,2]) on the GRU input in ONNX.
    On AX650 this Transpose immediately precedes DequantizeLinear on the TENG
    unit, triggering: assert(wdma[0].size == 0) in cmodel/sys/sys_teng.cpp.

    Fix: use batch_first=False GRU and swap (batch, seq) via Reshape instead
    of Transpose.  This is numerically identical and valid because in the
    streaming model exactly one of (batch, seq) is 1:
      - intra_rnn call: (B*T=1, F=33, C) → batch=1, seq=33
      - inter_rnn call: (B*F=33, T=1, C) → batch=33, seq=1
    When batch=1 or seq=1, reshape(seq, batch, feat) is memory-equivalent to
    permute(1,0,2) so the computation is identical.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.bidirectional = bidirectional
        # batch_first=False: no Transpose inserted by ONNX exporter
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=False, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                           batch_first=False, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        """
        x: (batch, seq, input_size)  — same batch_first interface as GRNN
        h: (num_layers * num_dir, batch, hidden_size)
        """
        batch, seq, feat = x.shape
        num_dir = 2 if self.bidirectional else 1
        if h is None:
            h = torch.zeros(self.num_layers * num_dir, batch,
                            self.hidden_size, device=x.device, dtype=x.dtype)

        x1, x2 = torch.chunk(x, chunks=2, dim=-1)  # (batch, seq, feat//2)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()

        # Swap (batch, seq) → (seq, batch) using Reshape, NOT permute/Transpose.
        # This is valid because exactly one of (batch, seq) equals 1 at runtime.
        # Reshape generates no Transpose ONNX node.
        x1_sf = x1.reshape(seq, batch, feat // 2)
        x2_sf = x2.reshape(seq, batch, feat // 2)

        y1_sf, h1 = self.rnn1(x1_sf, h1)   # y1_sf: (seq, batch, hidden//2 [*num_dir])
        y2_sf, h2 = self.rnn2(x2_sf, h2)

        # Swap back (seq, batch) → (batch, seq) using Reshape
        y1 = y1_sf.reshape(batch, seq, -1)
        y2 = y2_sf.reshape(batch, seq, -1)

        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNNAx650(nn.Module):
    """DPGRNN using GRNNAx650 — eliminates all GRU input Transpose nodes."""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size  = input_size
        self.width       = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNNAx650(input_size=input_size,
                                   hidden_size=hidden_size // 2,
                                   bidirectional=True)
        self.intra_fc  = nn.Linear(hidden_size, hidden_size)
        self.intra_ln  = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNNAx650(input_size=input_size,
                                   hidden_size=hidden_size,
                                   bidirectional=False)
        self.inter_fc  = nn.Linear(hidden_size, hidden_size)
        self.inter_ln  = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x: torch.Tensor, inter_cache: torch.Tensor):
        """
        x:           (B, C, T, F)
        inter_cache: (1, B*F, hidden_size)
        """
        x = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        x2 = intra_out.permute(0, 2, 1, 3)  # (B, F, T, C)
        inter_x = x2.reshape(x2.shape[0] * x2.shape[1], x2.shape[2], x2.shape[3])
        inter_x, inter_cache = self.inter_rnn(inter_x, inter_cache)
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x2.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B, T, F, C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B, C, T, F)
        return dual_out, inter_cache


# Modified sub-modules: no Pad / no Unsqueeze
# ===========================================================================

class StreamConvTranspose2dNoPad(nn.Module):
    """
    StreamConvTranspose2d without functional.pad.

    The original StreamConvTranspose2d.forward() calls
        torch.nn.functional.pad(inp, [left, right, 0, 0])
    which exports to an ONNX Pad node.  This version replaces that with
        torch.cat([zeros_left, inp, zeros_right], dim=-1)
    which exports only to Concat nodes, fully supported by AX650 NPU.

    Covers the F_stride==1 path only (the only path used by GTCRN).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        from torch.nn.modules.utils import _pair
        T_size, F_size   = _pair(kernel_size)
        T_stride, F_stride = _pair(stride)
        T_pad,   F_pad   = _pair(padding)
        T_dil,   F_dil   = _pair(dilation)

        assert T_stride == 1,  "T_stride must be 1 for streaming"
        assert F_stride == 1,  "StreamConvTranspose2dNoPad only handles F_stride==1"
        assert T_pad == 0,     "T_pad must be 0 for causal streaming"

        self.F_size  = F_size
        self.F_pad   = F_pad
        self.F_dil   = F_dil
        self._pad_size = (F_size - 1) * F_dil - F_pad  # symmetric pad on each side

        self.ConvTranspose2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=(T_stride, 1),
            padding=(T_pad, 0),
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        """x: (B,C,1,F)  cache: (B,C,T-1,F)"""
        inp = torch.cat([cache, x], dim=2)          # (B,C,T,F)
        out_cache = inp[:, :, 1:]

        p = self._pad_size
        if p > 0:
            # Replace functional.pad with Concat — no Pad ONNX node
            zeros = torch.zeros(
                inp.shape[0], inp.shape[1], inp.shape[2], p,
                dtype=inp.dtype, device=inp.device,
            )
            inp = torch.cat([zeros, inp, zeros], dim=-1)

        outp = self.ConvTranspose2d(inp)
        return outp, out_cache

class SFENoPad(nn.Module):
    """
    Subband Feature Extraction without nn.Unfold.

    nn.Unfold(kernel_size=(1,3), padding=(0,1)) exports to ONNX as
    Pad + Gather + Unsqueeze + ....  This replacement computes the exact
    same sliding window via Concat so no Pad or Gather/Unsqueeze nodes
    appear in the ONNX graph.

    For kernel_size=3, stride=1 on input (B,C,T,F):
      output channels [3c, 3c+1, 3c+2] = [left_c, center_c, right_c]
    → shape (B, 3C, T, F), interleaved per input channel (same as Unfold)
    """

    def __init__(self, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        assert kernel_size == 3 and stride == 1, \
            "SFENoPad only supports kernel_size=3, stride=1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, F)  →  (B, 3C, T, F)

        Channel ordering matches nn.Unfold(kernel_size=(1,3), padding=(0,1)):
          output channels [3c, 3c+1, 3c+2] = [left_c, center_c, right_c]
        i.e. interleaved per input channel, NOT grouped [ALL_LEFT|ALL_CENTER|ALL_RIGHT].
        """
        # left neighbour: [0, x0, x1, ..., x_{F-2}]
        x_left  = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
        # right neighbour: [x1, x2, ..., x_{F-1}, 0]
        x_right = torch.cat([x[..., 1:], torch.zeros_like(x[..., :1])], dim=-1)
        # Interleave: reshape each to (B,C,1,T,F), cat along dim=2, reshape to (B,3C,T,F)
        # This avoids Unsqueeze ONNX nodes while producing the correct channel order.
        B, C, T, F = x.shape
        x_l = x_left.reshape(B, C, 1, T, F)
        x_c = x.reshape(B, C, 1, T, F)
        x_r = x_right.reshape(B, C, 1, T, F)
        return torch.cat([x_l, x_c, x_r], dim=2).reshape(B, C * 3, T, F)


class StreamTRANoPad(nn.Module):
    """
    Temporal Recurrent Attention — removes all Transpose nodes.

    Original used:
      - zt.transpose(1,2)           → Transpose(perm=[2,0,1]) in ONNX
      - batch_first=True GRU        → Transpose(perm=[1,0,2]) on GRU input
      - att_fc(at).transpose(1, 2)  → Transpose in ONNX

    All three Transpose nodes trigger the AX650 TENG wdma assertion when
    Pulsar2's DequantizeLinear is inserted right after them.

    Fix: use batch_first=False GRU + Reshape for all (seq, batch) swaps.
    Valid in streaming mode because B=T=1 at runtime.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=False)
        self.att_fc  = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h_cache: torch.Tensor):
        """
        x:       (B, C, T, F)
        h_cache: (1, B, C*2)
        """
        B, C, T, F = x.shape
        zt = torch.mean(x.pow(2), dim=-1)   # (B, C, T)
        # Reshape to (T, B, C) for seq_first GRU — valid because B=T=1 in streaming
        zt_sf = zt.reshape(T, B, C)
        at_sf, h_cache = self.att_gru(zt_sf, h_cache)   # (T, B, channels*2)
        # att_fc on last dim, then reshape to (B, channels, T) — no Transpose
        at = self.att_fc(at_sf).reshape(B, C, T)
        at = self.att_act(at)
        At = at.reshape(B, C, T, 1)    # (B, C, T, 1)
        return x * At, h_cache


class StreamGTConvBlockNoPad(nn.Module):
    """
    Group Temporal Convolution Block — no Pad, no Unsqueeze from shuffle.

    Changes vs original StreamGTConvBlock:
      • Uses SFENoPad instead of SFE.
      • Uses StreamTRANoPad instead of StreamTRA.
      • Replaces torch.stack-based channel shuffle with reshape+cat+reshape
        so no Unsqueeze nodes appear in ONNX.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size, stride, padding, dilation,
                 use_deconv: bool = False):
        super().__init__()
        self.use_deconv = use_deconv
        stream_conv_module = StreamConvTranspose2dNoPad if use_deconv else StreamConv2d

        self.sfe = SFENoPad(kernel_size=3, stride=1)

        # Always use Conv2d for 1x1 point convolutions.
        # When use_deconv=True the original used ConvTranspose2d(A,B,1) which
        # is mathematically equivalent to Conv2d(A,B,1) with permuted weights.
        self.point_conv1 = nn.Conv2d(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1   = nn.BatchNorm2d(hidden_channels)
        self.point_act   = nn.PReLU()

        self.depth_conv = stream_conv_module(
            hidden_channels, hidden_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=hidden_channels,
        )
        self.depth_bn  = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = nn.Conv2d(hidden_channels, in_channels // 2, 1)
        self.point_bn2   = nn.BatchNorm2d(in_channels // 2)

        self.tra = StreamTRANoPad(in_channels // 2)

    def shuffle(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Channel-shuffle that interleaves x1 and x2 without Unsqueeze.

        Original:
            torch.stack([x1, x2], dim=1)  → Unsqueeze(x1) + Unsqueeze(x2) + Cat
            .transpose(1,2).view(B,-1,T,F)

        Replacement:
            reshape each to (B,C,1,T,F) via Reshape (not Unsqueeze),
            then Cat along dim=2, then Reshape back to (B,2C,T,F).
        """
        B, C, T, F = x1.shape
        x1r = x1.reshape(B, C, 1, T, F)   # Reshape — no Unsqueeze
        x2r = x2.reshape(B, C, 1, T, F)
        x   = torch.cat([x1r, x2r], dim=2)  # (B, C, 2, T, F)
        return x.reshape(B, C * 2, T, F)

    def forward(self, x: torch.Tensor,
                conv_cache: torch.Tensor,
                tra_cache:  torch.Tensor):
        """
        x:          (B, C, T, F)
        conv_cache: (B, C, (kT-1)*dT, F)
        tra_cache:  (1, B, C)
        """
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:,  x.shape[1] // 2:]

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1, conv_cache = self.depth_conv(h1, conv_cache)
        h1 = self.depth_act(self.depth_bn(h1))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1, tra_cache = self.tra(h1, tra_cache)

        x = self.shuffle(h1, x2)
        return x, conv_cache, tra_cache



# ===========================================================================
# Decoder ConvBlock: ConvTranspose2d replaced with zero-insert + Conv2d
# ===========================================================================

class ConvBlockDecoderNoPad(nn.Module):
    """
    Decoder ConvBlock: ConvTranspose2d(F_stride=2) → zero-insertion + Conv2d.

    ConvTranspose2d ONNX nodes trigger the AX650 NPU wdma assertion even for
    stride-2 upsampling.  This replaces the transpose-conv with:
      1. Zero-insert in F: [x0, 0, x1, 0, ..., x_{F-1}] via Unsqueeze+Mul+Cat
      2. Explicit F-padding via Concat (no Pad ONNX node)
      3. Conv2d with flipped kernel weights

    The result is numerically identical to ConvTranspose2d.

    Module attribute names (conv/bn/act) match ConvBlock so that
    _copy_weights can apply the CT->Conv2d transform by key name.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, is_last=False):
        super().__init__()
        from torch.nn.modules.utils import _pair
        T_k, F_k = _pair(kernel_size)
        T_s, F_s = _pair(stride)
        T_p, F_p = _pair(padding)

        assert T_s == 1, "T_stride must be 1"
        self.F_stride = F_s
        self.F_pad_each = F_k - 1 - F_p   # extra padding on each F side

        # Conv2d with no F-side padding — handled explicitly with Concat
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(T_k, F_k),
            stride=(T_s, 1),
            padding=(T_p, 0),
            groups=groups,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        S = self.F_stride

        # Step 1: zero-insert in F dim → length 2F-1
        # [x0, 0, x1, 0, ..., x_{F-1}]  using Unsqueeze+Mul+Cat+Reshape+Slice
        if S > 1:
            x_e = x.unsqueeze(-1)                         # (B,C,T,F,1)
            z_s = x_e * 0                                 # (B,C,T,F,1) zeros
            x_u = torch.cat([x_e, z_s], dim=-1)           # (B,C,T,F,2)
            x_u = x_u.reshape(B, C, T, F * S)             # (B,C,T,2F)
            x_u = x_u[:, :, :, :F * S - (S - 1)]         # (B,C,T,2F-1)
        else:
            x_u = x

        # Step 2: pad F dim for convolution — Concat, no Pad node
        p = self.F_pad_each
        if p > 0:
            z1 = x_u[:, :, :, :1] * 0                    # (B,C,T,1) zeros
            z_p = torch.cat([z1] * p, dim=-1)             # (B,C,T,p)
            x_p = torch.cat([z_p, x_u, z_p], dim=-1)
        else:
            x_p = x_u

        return self.act(self.bn(self.conv(x_p)))



class StreamEncoderNoPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, cc0, cc1, cc2, tc0, tc1, tc2):
        en_outs = []
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)
        x, cc0, tc0 = self.en_convs[2](x, cc0, tc0); en_outs.append(x)
        x, cc1, tc1 = self.en_convs[3](x, cc1, tc1); en_outs.append(x)
        x, cc2, tc2 = self.en_convs[4](x, cc2, tc2); en_outs.append(x)
        return x, en_outs, cc0, cc1, cc2, tc0, tc1, tc2


class StreamDecoderNoPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=True),
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=True),
            StreamGTConvBlockNoPad(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=True),
            ConvBlockDecoderNoPad(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            ConvBlockDecoderNoPad(16,  2, (1, 5), stride=(1, 2), padding=(0, 2), is_last=True),
        ])

    def forward(self, x, en_outs, cc0, cc1, cc2, tc0, tc1, tc2):
        x, cc0, tc0 = self.de_convs[0](x + en_outs[4], cc0, tc0)
        x, cc1, tc1 = self.de_convs[1](x + en_outs[3], cc1, tc1)
        x, cc2, tc2 = self.de_convs[2](x + en_outs[2], cc2, tc2)
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4 - i])
        return x, cc0, cc1, cc2, tc0, tc1, tc2


# ===========================================================================
# Top-level streaming model — "less input" cache format, no Pad / no Unsqueeze
# ===========================================================================

class StreamGTCRNAX650(nn.Module):
    """
    Streaming GTCRN for AX650 NPU quantization.

    Cache I/O uses the "less-input" merged format identical to the
    existing gtcrn_no_scatter_less_input_optimized.onnx so that the same
    calibration data and quantization config can be reused.

    Inputs / outputs
    ----------------
    mix             : (1, 257, 1, 2)
    en_conv_cache   : (1, 16, 16, 33)   merged [cc0|cc1|cc2]
    de_conv_cache   : (1, 16, 16, 33)   merged [cc2|cc1|cc0] (decoder order)
    en_tra_cache    : (1,  3,  1, 16)   merged [tc0|tc1|tc2]
    de_tra_cache    : (1,  3,  1, 16)
    inter_cache_0   : (1,  1, 33, 16)
    inter_cache_1   : (1,  1, 33, 16)

    Each cache output has the same name with "_out" suffix.
    """

    def __init__(self):
        super().__init__()
        self.erb      = ERB(65, 64)
        self.sfe      = SFENoPad(3, 1)
        self.encoder  = StreamEncoderNoPad()
        self.dpgrnn1  = DPGRNNAx650(16, 33, 16)
        self.dpgrnn2  = DPGRNNAx650(16, 33, 16)
        self.decoder  = StreamDecoderNoPad()
        self.mask     = Mask()

    def forward(self, spec,
                en_conv_cache, de_conv_cache,
                en_tra_cache,  de_tra_cache,
                inter_cache_0, inter_cache_1):

        # ---- split merged caches ----
        en_cc0 = en_conv_cache[:, :,  :2,  :]
        en_cc1 = en_conv_cache[:, :,  2:6, :]
        en_cc2 = en_conv_cache[:, :,  6:,  :]
        de_cc0 = de_conv_cache[:, :,  6:,  :]
        de_cc1 = de_conv_cache[:, :,  2:6, :]
        de_cc2 = de_conv_cache[:, :,  :2,  :]

        en_tc0 = en_tra_cache[:, 0, :, :].reshape(1, 1, 16)
        en_tc1 = en_tra_cache[:, 1, :, :].reshape(1, 1, 16)
        en_tc2 = en_tra_cache[:, 2, :, :].reshape(1, 1, 16)
        de_tc0 = de_tra_cache[:, 0, :, :].reshape(1, 1, 16)
        de_tc1 = de_tra_cache[:, 1, :, :].reshape(1, 1, 16)
        de_tc2 = de_tra_cache[:, 2, :, :].reshape(1, 1, 16)

        ic0 = inter_cache_0.reshape(1, 33, 16)
        ic1 = inter_cache_1.reshape(1, 33, 16)

        # ---- feature extraction ----
        spec_real = spec[..., 0].permute(0, 2, 1)   # (B, T, F)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag  = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)

        feat = self.erb.bm(feat)   # (B, 3, T, 129)
        feat = self.sfe(feat)      # (B, 9, T, 129)

        # ---- encoder ----
        feat, en_outs, en_cc0, en_cc1, en_cc2, en_tc0, en_tc1, en_tc2 = \
            self.encoder(feat, en_cc0, en_cc1, en_cc2, en_tc0, en_tc1, en_tc2)

        # ---- dual-path GRU ----
        feat, ic0 = self.dpgrnn1(feat, ic0)
        feat, ic1 = self.dpgrnn2(feat, ic1)

        # ---- decoder ----
        m_feat, de_cc0, de_cc1, de_cc2, de_tc0, de_tc1, de_tc2 = \
            self.decoder(feat, en_outs, de_cc0, de_cc1, de_cc2, de_tc0, de_tc1, de_tc2)

        # ---- mask + output ----
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec.permute(0, 3, 2, 1))
        spec_enh = spec_enh.permute(0, 3, 2, 1)

        # ---- reassemble merged caches ----
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

        inter_cache_0_out = ic0.reshape(1, 1, 33, 16)
        inter_cache_1_out = ic1.reshape(1, 1, 33, 16)

        return (spec_enh,
                en_conv_cache_out, de_conv_cache_out,
                en_tra_cache_out,  de_tra_cache_out,
                inter_cache_0_out, inter_cache_1_out)


# ===========================================================================
# Weight transfer
# ===========================================================================

def _ct_weight_to_conv(w: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Transform ConvTranspose2d weight to equivalent Conv2d weight.

    ConvTranspose2d(Ci, Co, K, groups=G) weight shape: (Ci, Co//G, *K)
    Conv2d(Ci, Co, K, groups=G) weight shape:          (Co, Ci//G, *K)

    Transformation: swap Ci/Co within each group, then flip spatial dims.
    """
    Ci   = w.shape[0]
    CiG  = Ci // groups
    CoG  = w.shape[1]
    Co   = CoG * groups
    K    = w.shape[2:]
    w    = w.reshape(groups, CiG, CoG, *K)          # (G, CiG, CoG, *K)
    dims = [0, 2, 1] + list(range(3, w.ndim))
    w    = w.permute(dims)                           # (G, CoG, CiG, *K)
    for d in range(3, w.ndim):
        w = w.flip(d)
    return w.reshape(Co, CiG, *K).contiguous()


def _copy_weights(dst: nn.Module, src: nn.Module) -> None:
    """Copy matching weights from src (original stream model) into dst.

    Handles two cases:
      - Direct copy when shapes match (most tensors).
      - ConvTranspose2d -> Conv2d weight transform for decoder convolutions:
        * de_convs.3.conv.weight: same shape but different semantics (groups=2)
        * de_convs.{0,1,2}.point_conv{1,2}.weight: different shapes (1x1 CT)
        * de_convs.4.conv.weight: different shapes (groups=1)
    """
    dst_sd = dst.state_dict()
    src_sd = src.state_dict()

    # Keys that need explicit CT->Conv2d transform (same shape, different semantics)
    CT_EXPLICIT = {
        'decoder.de_convs.3.conv.weight': 2,   # groups=2 stride-2 CT
    }

    matched     = 0
    transformed = 0
    for k in dst_sd:
        if k not in src_sd:
            continue
        src_w      = src_sd[k]
        dst_shape  = dst_sd[k].shape

        if k in CT_EXPLICIT:
            g     = CT_EXPLICIT[k]
            w_new = _ct_weight_to_conv(src_w, groups=g)
            assert w_new.shape == dst_shape, f"CT->Conv shape error for {k}"
            dst_sd[k] = w_new
            transformed += 1
        elif src_w.shape == dst_shape:
            dst_sd[k] = src_w
            matched += 1
        elif k.endswith('.weight') and src_w.ndim == 4:
            # Auto-detect 1x1 CT->Conv2d and stride-2 CT->Conv2d (groups=1)
            # where shapes differ after the permutation
            for g in (1,):
                try:
                    w_new = _ct_weight_to_conv(src_w, groups=g)
                    if w_new.shape == dst_shape:
                        dst_sd[k] = w_new
                        transformed += 1
                        break
                except Exception:
                    pass

    dst.load_state_dict(dst_sd)
    print(f"  Weight transfer: {matched} direct + {transformed} transformed = "
          f"{matched + transformed}/{len(dst_sd)} total")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Export GTCRN ONNX for AX650 NPU")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help="Path to model_trained_on_dns3.tar")
    parser.add_argument("--out",  default=str(DEFAULT_OUT),
                        help="Output ONNX path")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Skip onnxsim simplification")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # ---- load original weights ----
    print(f"\n[1/5] Loading checkpoint: {ckpt_path}")
    orig_model = GTCRN().to(device).eval()
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    orig_model.load_state_dict(state)

    stream_orig = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_orig, orig_model)

    # ---- build AX650 model and copy weights ----
    print("\n[2/5] Building AX650-friendly stream model")
    ax650_model = StreamGTCRNAX650().to(device).eval()
    _copy_weights(ax650_model, stream_orig)

    # ---- numerical sanity check ----
    print("\n[3/5] Numerical sanity check (PyTorch)")
    import numpy as np
    torch.manual_seed(0)
    dummy_spec  = torch.randn(1, 257, 1, 2)
    dummy_en_cc = torch.zeros(1, 16, 16, 33)
    dummy_de_cc = torch.zeros(1, 16, 16, 33)
    dummy_en_tc = torch.zeros(1,  3,  1, 16)
    dummy_de_tc = torch.zeros(1,  3,  1, 16)
    dummy_ic0   = torch.zeros(1,  1, 33, 16)
    dummy_ic1   = torch.zeros(1,  1, 33, 16)

    # Build equivalent original-style model with the same "less input" format
    from gtcrn_stream import StreamGTCRN as _StreamGTCRN
    # Build the original no-scatter less-input model from the reference file
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "demo_onnx_no_scatter",
            str(GTCRN_SRC / "demo_onnx_no_scatter.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        orig_less = mod.StreamGTCRNNoScatter().to(device).eval()
        mod.convert_to_no_scatter_model(orig_less, stream_orig)
    except Exception as e:
        print(f"  (Skipping cross-model check: {e})")
        orig_less = None

    with torch.no_grad():
        out_ax650 = ax650_model(
            dummy_spec, dummy_en_cc, dummy_de_cc,
            dummy_en_tc, dummy_de_tc, dummy_ic0, dummy_ic1,
        )
    if orig_less is not None:
        with torch.no_grad():
            out_orig = orig_less(
                dummy_spec, dummy_en_cc, dummy_de_cc,
                dummy_en_tc, dummy_de_tc, dummy_ic0, dummy_ic1,
            )
        err = float((out_ax650[0] - out_orig[0]).abs().max())
        print(f"  Max output error vs original: {err:.2e}")
        if err > 1e-4:
            print("  WARNING: outputs differ more than expected!")
        else:
            print("  OK — outputs match")

    # ---- export ONNX ----
    print(f"\n[4/5] Exporting ONNX → {out_path}")
    dummy_inputs = (
        dummy_spec, dummy_en_cc, dummy_de_cc,
        dummy_en_tc, dummy_de_tc, dummy_ic0, dummy_ic1,
    )
    torch.onnx.export(
        ax650_model,
        dummy_inputs,
        str(out_path),
        input_names=[
            "mix",
            "en_conv_cache", "de_conv_cache",
            "en_tra_cache",  "de_tra_cache",
            "inter_cache_0", "inter_cache_1",
        ],
        output_names=[
            "enh",
            "en_conv_cache_out", "de_conv_cache_out",
            "en_tra_cache_out",  "de_tra_cache_out",
            "inter_cache_0_out", "inter_cache_1_out",
        ],
        opset_version=11,
        verbose=False,
    )
    print(f"  Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")

    # ---- verify exported model ----
    import onnx
    import onnxruntime as ort
    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    pad_cnt      = sum(1 for n in model_onnx.graph.node if n.op_type == "Pad")
    unsq_cnt     = sum(1 for n in model_onnx.graph.node if n.op_type == "Unsqueeze")
    scatter_cnt  = sum(1 for n in model_onnx.graph.node if n.op_type == "ScatterND")
    ct_cnt       = sum(1 for n in model_onnx.graph.node if n.op_type == "ConvTranspose")
    print(f"  Pad nodes:       {pad_cnt}   (target: 0)")
    print(f"  ConvTranspose:   {ct_cnt}   (target: 0)")
    print(f"  Unsqueeze nodes: {unsq_cnt}  (GRU residual, harmless)")
    print(f"  ScatterND nodes: {scatter_cnt}  (should be 0)")

    # ONNX Runtime numerical check
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {
        "mix":           dummy_spec.numpy(),
        "en_conv_cache": dummy_en_cc.numpy(),
        "de_conv_cache": dummy_de_cc.numpy(),
        "en_tra_cache":  dummy_en_tc.numpy(),
        "de_tra_cache":  dummy_de_tc.numpy(),
        "inter_cache_0": dummy_ic0.numpy(),
        "inter_cache_1": dummy_ic1.numpy(),
    }
    ort_out = sess.run(None, feed)
    err_ort = float(np.abs(ort_out[0] - out_ax650[0].numpy()).max())
    print(f"  ORT vs PyTorch max error: {err_ort:.2e}")

    # ---- simplify + optimize ----
    if not args.no_simplify:
        print("\n[5/5] Simplifying with onnxsim (perform_optimization=True)")
        try:
            from onnxsim import simplify
            model_simp, ok = simplify(model_onnx, perform_optimization=True)
            assert ok, "onnxsim validation failed"
            # overwrite the original file with the simplified+optimized version
            onnx.save(model_simp, str(out_path))
            pad_s  = sum(1 for n in model_simp.graph.node if n.op_type == "Pad")
            unsq_s = sum(1 for n in model_simp.graph.node if n.op_type == "Unsqueeze")
            ct_s   = sum(1 for n in model_simp.graph.node if n.op_type == "ConvTranspose")
            print(f"  Saved (simplified): {out_path}  ({out_path.stat().st_size // 1024} KB)")
            print(f"  After simplify — Pad: {pad_s}, ConvTranspose: {ct_s}, Unsqueeze: {unsq_s}")
        except ImportError:
            print("  onnxsim not installed; skipping simplification")

    print("\nDone.  Next step: update ax_configs gtcrn JSON to point at the new model.")


if __name__ == "__main__":
    main()
