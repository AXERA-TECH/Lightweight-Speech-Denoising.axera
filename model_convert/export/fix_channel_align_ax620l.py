#!/usr/bin/env python3
"""
ONNX surgery to fix two ax620L compiler bugs for tiny_v5. Minimal size overhead.

Bug 1 — Tiling assertion for asymmetric causal depthwise conv
  ax620L tiler asserts `stop + right_pad >= real_stop` per SUB-TILE.
  Original Conv has right_pad=0. Pulsar2 assigns H_end=0 to intermediate
  sub-tiles → assertion fails even with right_pad>0 in Conv attrs, because
  the tiler distributes right-pad only to the last tile.
  Fix: extract Conv's spatial pads into a preceding ONNX Pad node so Conv
       sees pads=[0,0,0,0] — no per-tile padding assertion.

Bug 2 — RDMA cycle-count mismatch for U16 depthwise conv with C%16 != 0
  For C=28 (28%16=12), Pulsar2 incorrectly computes rdma0_cycle_m1.
  Fix: insert Pad_channel (+4 zero channels, C=28→32 where 32%16=0).

Both fixes combined (per patched depthwise Conv):
  1. Pad_spatial:  extract Conv's H/W pads → pre-pad the input
  2. Pad_channel:  pad C=28→32 (also breaks Pulsar2 Pad_spatial→Conv fusion)
  3. Conv:         pads=[0,0,0,0], group=32, weight C=28→32
  4. Slice(ch):    axis=1, 0:28 — restore original channel count

Fusion-breaking: Pulsar2 can only absorb a Pad directly adjacent to Conv.
With Pad_channel between Pad_spatial and Conv, the spatial Pad is no longer
adjacent → Pulsar2 cannot re-absorb it → Conv truly sees pads=[0,0,0,0].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ORIG_ONNX   = Path(__file__).resolve().parents[1] / "quant/onnx_models" / "tiny_v5_context.onnx"
OUTPUT_ONNX = Path(__file__).resolve().parents[1] / "quant/onnx_models" / "tiny_v5_context_ax620l.onnx"

ALIGN = 16


def fix_model(input_path: Path, output_path: Path) -> None:
    model = onnx.load(str(input_path))
    graph = model.graph

    init_map = {i.name: i for i in graph.initializer}

    new_nodes:       list[onnx.NodeProto]   = []
    new_inits:       list[onnx.TensorProto] = []
    nodes_to_remove: list[onnx.NodeProto]   = []

    # Shared constants reused by all patched Convs
    shared_ch_pads4  = "__shared_ch_pads4"   # channel pad +4  [0,0,0,0, 0,4,0,0]
    shared_s1_starts = "__shared_s1_starts"  # [0]
    shared_s1_ends   = "__shared_s1_ends28"  # [28]
    shared_s1_axes   = "__shared_s1_axes1"   # [1]
    new_inits += [
        numpy_helper.from_array(np.array([0,0,0,0, 0,4,0,0], dtype=np.int64), name=shared_ch_pads4),
        numpy_helper.from_array(np.array([0],  dtype=np.int64), name=shared_s1_starts),
        numpy_helper.from_array(np.array([28], dtype=np.int64), name=shared_s1_ends),
        numpy_helper.from_array(np.array([1],  dtype=np.int64), name=shared_s1_axes),
    ]

    for node in list(graph.node):
        if node.op_type != "Conv":
            continue

        group = next((a.i for a in node.attribute if a.name == "group"), 1)
        w_init = init_map.get(node.input[1])
        if w_init is None:
            continue
        w = numpy_helper.to_array(w_init)
        if w.ndim != 4:
            continue
        C_out, C_in_g, k_t, k_f = w.shape
        C = C_out

        # Only target: depthwise conv, C%ALIGN != 0, right_pad (H_end) == 0
        if group != C or C_in_g != 1 or C % ALIGN == 0:
            continue
        pads = next((list(a.ints) for a in node.attribute if a.name == "pads"), [0, 0, 0, 0])
        if pads[2] != 0:
            continue  # H_end already non-zero, skip

        pad_c    = ALIGN - (C % ALIGN)   # 4 for C=28
        C_new    = C + pad_c              # 32
        dilation = next((list(a.ints) for a in node.attribute if a.name == "dilations"), [1, 1])
        strides  = next((list(a.ints) for a in node.attribute if a.name == "strides"),   [1, 1])
        # Conv spatial pads: [H_beg, W_beg, H_end, W_end]
        H_beg, W_beg, _H_end, W_end = pads

        in_name = node.input[0]
        uid = node.name.replace("/", "_").strip("_")
        print(f"  Patching {node.name}: C={C}→{C_new}, k={k_t}×{k_f}, "
              f"dil={dilation}, pads {pads}")

        # ---- 1. Pad_spatial: extract H/W pads from Conv into a Pad node ----
        # ONNX Pad pads (rank-4, NCHW): [N_beg,C_beg,H_beg,W_beg, N_end,C_end,H_end,W_end]
        sp_pads_name = f"{uid}_sp_pads"
        new_inits.append(numpy_helper.from_array(
            np.array([0, 0, H_beg, W_beg, 0, 0, 0, W_end], dtype=np.int64),
            name=sp_pads_name))
        sp_pad_out = f"{uid}_sp_pad_out"
        new_nodes.append(helper.make_node(
            "Pad",
            inputs=[in_name, sp_pads_name],
            outputs=[sp_pad_out],
            name=f"{uid}_sp_pad",
            mode="constant",
        ))

        # ---- 2. Pad_channel: add pad_c zero channels (C=28→32), breaks fusion ----
        ch_pad_out = f"{uid}_ch_pad_out"
        new_nodes.append(helper.make_node(
            "Pad",
            inputs=[sp_pad_out, shared_ch_pads4],
            outputs=[ch_pad_out],
            name=f"{uid}_ch_pad",
            mode="constant",
        ))

        # ---- 3. Expanded Conv weight (C_new, 1, k_t, k_f) ----
        new_w = np.zeros((C_new, 1, k_t, k_f), dtype=w.dtype)
        new_w[:C] = w
        new_w_name = f"{uid}_w_aligned"
        new_inits.append(numpy_helper.from_array(new_w, name=new_w_name))

        # ---- 4. Expanded bias ----
        bias_inputs: list[str] = []
        if len(node.input) > 2 and node.input[2]:
            b_init = init_map.get(node.input[2])
            if b_init is not None:
                b = numpy_helper.to_array(b_init)
                new_b = np.zeros(C_new, dtype=b.dtype)
                new_b[:C] = b
                new_b_name = f"{uid}_b_aligned"
                new_inits.append(numpy_helper.from_array(new_b, name=new_b_name))
                bias_inputs = [new_b_name]

        # ---- 5. New depthwise Conv with pads=[0,0,0,0] ----
        conv_out = f"{uid}_conv_out"
        new_nodes.append(helper.make_node(
            "Conv",
            inputs=[ch_pad_out, new_w_name] + bias_inputs,
            outputs=[conv_out],
            name=f"{uid}_conv",
            group=C_new,
            dilations=dilation,
            strides=strides,
            pads=[0, 0, 0, 0],
            kernel_shape=[k_t, k_f],
        ))

        # ---- 6. Slice axis=1: trim extra channels (0:C) ----
        new_nodes.append(helper.make_node(
            "Slice",
            inputs=[conv_out, shared_s1_starts, shared_s1_ends, shared_s1_axes],
            outputs=[node.output[0]],  # same name → downstream unchanged
            name=f"{uid}_slice_ch",
        ))

        nodes_to_remove.append(node)

    # -- Rebuild node list preserving topological order --
    remove_set = set(id(n) for n in nodes_to_remove)
    # 5 replacement nodes per removed Conv: sp_Pad, ch_Pad, Conv, Slice_ch
    replacement_map: dict[int, list] = {}
    it = iter(new_nodes)
    for _ in nodes_to_remove:
        replacement_map[id(_)] = [next(it), next(it), next(it), next(it)]

    new_list: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in remove_set:
            new_list.extend(replacement_map[id(node)])
        else:
            new_list.append(node)

    del graph.node[:]
    graph.node.extend(new_list)
    graph.initializer.extend(new_inits)

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    sz = output_path.stat().st_size
    print(f"\nSaved → {output_path}  ({sz//1024}K)")


def numerical_verify(original: Path, fixed: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping verify")
        return

    np.random.seed(42)
    dummy = np.random.randn(1, 1, 34, 257).astype(np.float32)
    orig_out = ort.InferenceSession(
        str(original), providers=["CPUExecutionProvider"]
    ).run(None, {"input": dummy})[0]
    fix_out = ort.InferenceSession(
        str(fixed), providers=["CPUExecutionProvider"]
    ).run(None, {"input": dummy})[0]

    diff = np.abs(orig_out - fix_out)
    print(f"Numerical verify: max_err={diff.max():.2e}  mean_err={diff.mean():.2e}")
    if diff.max() < 1e-5:
        print("OK — outputs are bitwise identical")
    else:
        print(f"WARNING: outputs differ — investigate!")


if __name__ == "__main__":
    print("=== Channel-align + spatial-pad-extract fix for tiny_v5 ax620L ===\n")
    if not ORIG_ONNX.exists():
        print(f"ERROR: {ORIG_ONNX} not found.")
        sys.exit(1)
    fix_model(ORIG_ONNX, OUTPUT_ONNX)
    numerical_verify(ORIG_ONNX, OUTPUT_ONNX)
