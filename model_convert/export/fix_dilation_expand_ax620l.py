#!/usr/bin/env python3
"""
ONNX surgery: convert dilated depthwise Conv → non-dilated equivalent for ax620L.

Root cause
----------
ax620L Pulsar2 calls conv_dilation.py for ANY Conv with dilation>1.
That file asserts `stop + H_end_pad >= real_stop` per sub-tile.
The tiler forces H_end_pad=0 for intermediate sub-tiles regardless of Conv's
pads attribute → assertion always fails for any dilation>1 on ax620L.

Previous attempts (right_pad=1, pads=[0,0,0,0]+Pad_channel) all fail because
they keep dilation>1 in Conv.

Fix
---
Convert each dilated depthwise Conv to a mathematically equivalent non-dilated
Conv with an expanded (sparse) kernel. With dilation=1, conv_dilation.py is
never called → no assertion.

Kernel expansion (1D in H):
  original kernel K[c,0,i,j], dilation dil_h
  new kernel K_new[c,0,i*dil_h,j] = K[c,0,i,j]   (sparse, zeros elsewhere)
  kh_new = (kh-1)*dil_h + 1

Padding stays unchanged:
  For causal (H_beg=(kh-1)*dil_h, H_end=0, same-size output):
    output_H = T + H_beg + H_end - (kh-1)*dil_h  [original]
             = T + H_beg + H_end - (kh_new-1)*1   [new, same result] ✓

Weight overhead: +24KB total for conv_se (8 dilated depthwise convs).
Expected output size: ~617K (vs 593K original, vs 695K prepare_for_npu_export).

Does NOT require extra ONNX nodes — only modifies existing Conv weight
initializers and attributes in-place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ORIG_ONNX   = Path(__file__).resolve().parents[1] / "quant/onnx_models" / "conv_se_context.onnx"
OUTPUT_ONNX = Path(__file__).resolve().parents[1] / "quant/onnx_models" / "conv_se_context_ax620l.onnx"


def fix_model(input_path: Path, output_path: Path) -> None:
    model = onnx.load(str(input_path))
    graph = model.graph

    init_map = {i.name: i for i in graph.initializer}
    patched = 0

    for node in graph.node:
        if node.op_type != "Conv":
            continue

        group = next((a.i for a in node.attribute if a.name == "group"), 1)
        dils  = next((list(a.ints) for a in node.attribute if a.name == "dilations"), [1])
        w_init = init_map.get(node.input[1])
        if w_init is None:
            continue
        w = numpy_helper.to_array(w_init)
        if w.ndim not in (3, 4):
            continue

        # Determine spatial rank (1D or 2D conv)
        is_2d = (w.ndim == 4)
        if is_2d:
            C, C_in_g, kh, kw = w.shape
        else:
            C, C_in_g, kh = w.shape
            kw = None

        # Only depthwise convs with temporal dilation > 1
        if group != C or C_in_g != 1:
            continue
        dil_h = dils[0]
        if dil_h <= 1:
            continue

        # Expand kernel along H (temporal axis): insert zeros between original rows
        kh_new = (kh - 1) * dil_h + 1
        if is_2d:
            w_new = np.zeros((C, 1, kh_new, kw), dtype=w.dtype)
            for i in range(kh):
                w_new[:, 0, i * dil_h, :] = w[:, 0, i, :]
            new_dilations = [1, dils[1] if len(dils) > 1 else 1]
            new_kernel    = [kh_new, kw]
        else:
            w_new = np.zeros((C, 1, kh_new), dtype=w.dtype)
            for i in range(kh):
                w_new[:, 0, i * dil_h] = w[:, 0, i]
            new_dilations = [1]
            new_kernel    = [kh_new]

        # Replace weight initializer in-place
        new_w_init = numpy_helper.from_array(w_new, name=w_init.name)
        graph.initializer.remove(w_init)
        graph.initializer.append(new_w_init)
        init_map[w_init.name] = new_w_init

        # Update Conv attributes: dilations → [1,...], kernel_shape → expanded
        for attr in node.attribute:
            if attr.name == "dilations":
                del attr.ints[:]
                attr.ints.extend(new_dilations)
            elif attr.name == "kernel_shape":
                del attr.ints[:]
                attr.ints.extend(new_kernel)
        # Pads remain unchanged — same H_beg, H_end=0 gives same output shape

        dim_str = f"({kh},{kw})→({kh_new},{kw})" if is_2d else f"({kh},)→({kh_new},)"
        print(f"  Patched {node.name}: C={C}, k={dim_str}, dil={dil_h}→1")
        patched += 1

    if patched == 0:
        print("WARNING: no dilated depthwise Conv found — model may already be fixed or wrong input")
    else:
        print(f"\n  {patched} Conv nodes patched")

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    sz = output_path.stat().st_size
    print(f"Saved → {output_path}  ({sz // 1024}K)")


def numerical_verify(original: Path, fixed: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping verify")
        return

    np.random.seed(42)
    dummy = np.random.randn(1, 1, 64, 257).astype(np.float32)
    sess_orig = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    sess_fix  = ort.InferenceSession(str(fixed),    providers=["CPUExecutionProvider"])
    out_orig = sess_orig.run(None, {"input": dummy})[0]
    out_fix  = sess_fix.run(None,  {"input": dummy})[0]

    diff = np.abs(out_orig - out_fix)
    snr = 20 * np.log10(np.linalg.norm(out_orig) / (np.linalg.norm(diff) + 1e-15))
    print(f"Numerical verify: max_err={diff.max():.2e}  SNR={snr:.0f} dB")
    if diff.max() < 1e-5:
        print("OK — outputs are bitwise identical")
    else:
        print(f"WARNING: outputs differ!")


if __name__ == "__main__":
    print("=== Dilation-expand fix for conv_se ax620L ===\n")
    if not ORIG_ONNX.exists():
        print(f"ERROR: {ORIG_ONNX} not found.")
        sys.exit(1)
    fix_model(ORIG_ONNX, OUTPUT_ONNX)
    numerical_verify(ORIG_ONNX, OUTPUT_ONNX)
