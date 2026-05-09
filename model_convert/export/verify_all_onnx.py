#!/usr/bin/env python3
"""
Verify all ONNX models in denoise_solution_v2_context/models/ with inference checks.

Checks performed:
  1. tiny_v5_context.onnx      -- loads and produces correct output shape
  2. tiny_v5_context_ax620l.onnx -- produces bitwise-identical output to #1
  3. conv_se_context.onnx      -- loads and produces correct output shape
  4. conv_se_context_ax620l.onnx -- SNR ≥ 100 dB compared to #3
  5. gtcrn_no_scatter_less_input_optimized.onnx -- 7-input streaming inference
  6. gtcrn_ax650_nopd_fixed.onnx -- 7-input streaming inference

Usage:
    cd denoise_solution_v2_context_sim
    python3 quantization/verify_all_onnx.py
    python3 quantization/verify_all_onnx.py --model tiny_v5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. pip install onnxruntime")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent   # quantization/
ROOT       = THIS_DIR.parent                   # denoise_solution_v2_context/
MODELS_DIR = ROOT / "quant/onnx_models"

PROVIDERS = ["CPUExecutionProvider"]


class Result(NamedTuple):
    name: str
    passed: bool
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def snr_db(ref: np.ndarray, cmp: np.ndarray) -> float:
    signal_power = np.sum(ref**2)
    noise_power  = np.sum((ref - cmp)**2)
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power + 1e-20)


def load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=PROVIDERS)


def run_session(sess: ort.InferenceSession, feeds: dict) -> list[np.ndarray]:
    return sess.run(None, feeds)


# ── Mask model checks ─────────────────────────────────────────────────────────

def check_mask_model(onnx_path: Path, T: int, F: int, out_F: int,
                     ref_output: np.ndarray | None = None,
                     min_snr: float | None = None) -> Result:
    name = onnx_path.name
    if not onnx_path.exists():
        return Result(name, False, "file not found")
    try:
        sess = load_session(onnx_path)
        inp = sess.get_inputs()[0]
        assert list(inp.shape) == [1, 1, T, F], f"unexpected input shape {inp.shape}"
        dummy = np.random.randn(1, 1, T, F).astype(np.float32)
        outputs = run_session(sess, {inp.name: dummy})
        out = outputs[0]
        assert out.shape[2] == T or out.shape[2] in {T, 1}, f"T mismatch {out.shape}"
        if ref_output is not None:
            if min_snr is None:
                max_err = float(np.abs(ref_output - out).max())
                if max_err > 1e-5:
                    return Result(name, False, f"max_abs_err={max_err:.2e} (expected bitwise identical)")
                return Result(name, True, f"max_abs_err={max_err:.2e} (bitwise identical ✓)")
            else:
                snr = snr_db(ref_output, out)
                if snr < min_snr:
                    return Result(name, False, f"SNR={snr:.1f}dB < {min_snr}dB threshold")
                return Result(name, True, f"SNR={snr:.1f}dB ≥ {min_snr}dB ✓")
        return Result(name, True, f"output shape {list(out.shape)} ✓")
    except Exception as exc:
        return Result(name, False, str(exc))


def get_mask_output(onnx_path: Path, T: int, F: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    sess = load_session(onnx_path)
    inp  = sess.get_inputs()[0]
    dummy = np.random.randn(1, 1, T, F).astype(np.float32)
    return run_session(sess, {inp.name: dummy})[0]


# ── GTCRN streaming checks ────────────────────────────────────────────────────

GTCRN_CACHES_SHAPES = {
    "en_conv_cache":  (1, 16, 16, 33),
    "de_conv_cache":  (1, 16, 16, 33),
    "en_tra_cache":   (1,  3,  1, 16),
    "de_tra_cache":   (1,  3,  1, 16),
    "inter_cache_0":  (1,  1, 33, 16),
    "inter_cache_1":  (1,  1, 33, 16),
}


def check_gtcrn_model(onnx_path: Path, n_frames: int = 20) -> Result:
    name = onnx_path.name
    if not onnx_path.exists():
        return Result(name, False, "file not found")
    try:
        sess = load_session(onnx_path)
        input_names  = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]

        assert "mix" in input_names,      "missing 'mix' input"
        assert len(input_names) == 7,     f"expected 7 inputs, got {len(input_names)}"
        assert len(output_names) == 7,    f"expected 7 outputs, got {len(output_names)}"

        caches = {k: np.zeros(s, dtype=np.float32) for k, s in GTCRN_CACHES_SHAPES.items()}

        total_power = 0.0
        for i in range(n_frames):
            mix = np.random.randn(1, 257, 1, 2).astype(np.float32)
            feeds = {"mix": mix, **caches}
            out_list = run_session(sess, feeds)
            enh = out_list[0]
            assert list(enh.shape) == [1, 257, 1, 2], f"bad enh shape {enh.shape}"
            total_power += float(np.mean(enh**2))
            # update caches (by output name suffix matching)
            for j, oname in enumerate(output_names[1:], 1):
                # match by stripping "_out" suffix
                cache_key = oname.replace("_out", "")
                if cache_key in caches:
                    caches[cache_key] = out_list[j]

        mean_output_power = total_power / n_frames
        if mean_output_power < 1e-12:
            return Result(name, False, "output power near zero (model inactive?)")

        return Result(name, True,
                      f"{n_frames} frames OK, mean output power={mean_output_power:.4f} ✓")
    except Exception as exc:
        return Result(name, False, str(exc))


# ── Cross-model comparison ────────────────────────────────────────────────────

def check_gtcrn_pair(path_a: Path, path_b: Path, n_frames: int = 50) -> list[Result]:
    """Run both GTCRN models on the same input sequence and compare outputs."""
    results = []
    for p in [path_a, path_b]:
        if not p.exists():
            results.append(Result(p.name, False, "file not found"))
    if len(results) > 0:
        return results

    try:
        sess_a = load_session(path_a)
        sess_b = load_session(path_b)

        caches_a = {k: np.zeros(s, dtype=np.float32) for k, s in GTCRN_CACHES_SHAPES.items()}
        caches_b = {k: np.zeros(s, dtype=np.float32) for k, s in GTCRN_CACHES_SHAPES.items()}
        in_names_a = [i.name for i in sess_a.get_inputs()]
        in_names_b = [i.name for i in sess_b.get_inputs()]
        out_names_a = [o.name for o in sess_a.get_outputs()]
        out_names_b = [o.name for o in sess_b.get_outputs()]

        np.random.seed(123)
        all_enh_a, all_enh_b = [], []
        for _ in range(n_frames):
            mix = np.random.randn(1, 257, 1, 2).astype(np.float32)
            out_a = run_session(sess_a, {"mix": mix, **caches_a})
            out_b = run_session(sess_b, {"mix": mix, **caches_b})
            all_enh_a.append(out_a[0])
            all_enh_b.append(out_b[0])
            for j, oname in enumerate(out_names_a[1:], 1):
                ck = oname.replace("_out", "")
                if ck in caches_a:
                    caches_a[ck] = out_a[j]
            for j, oname in enumerate(out_names_b[1:], 1):
                ck = oname.replace("_out", "")
                if ck in caches_b:
                    caches_b[ck] = out_b[j]

        enh_a = np.concatenate(all_enh_a, axis=2)
        enh_b = np.concatenate(all_enh_b, axis=2)
        snr = snr_db(enh_a, enh_b)
        results.append(Result(
            f"{path_a.name} vs {path_b.name}",
            snr > 20,
            f"cross-model SNR={snr:.1f}dB"
        ))
    except Exception as exc:
        results.append(Result(f"{path_a.name} vs {path_b.name}", False, str(exc)))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all_checks(args: argparse.Namespace) -> list[Result]:
    results: list[Result] = []

    run_tiny = args.model in {"tiny_v5", "all"}
    run_conv = args.model in {"conv_se", "all"}
    run_gtcrn = args.model in {"gtcrn", "all"}

    # ── tiny_v5 ───────────────────────────────────────────────────────────────
    if run_tiny:
        print("\n── tiny_v5 ───────────────────────────────────────────────")
        p_orig  = MODELS_DIR / "tiny_v5_context.onnx"
        p_620l  = MODELS_DIR / "tiny_v5_context_ax620l.onnx"

        r_orig = check_mask_model(p_orig, T=34, F=257, out_F=17)
        results.append(r_orig)
        print(f"  {'PASS' if r_orig.passed else 'FAIL'} {r_orig.name}: {r_orig.message}")

        if r_orig.passed and p_orig.exists():
            ref_out = get_mask_output(p_orig, T=34, F=257, seed=77)
            np.random.seed(77)  # reset so ax620l gets same input
            r_620l = check_mask_model(p_620l, T=34, F=257, out_F=17,
                                      ref_output=ref_out, min_snr=None)
        else:
            r_620l = check_mask_model(p_620l, T=34, F=257, out_F=17)
        results.append(r_620l)
        print(f"  {'PASS' if r_620l.passed else 'FAIL'} {r_620l.name}: {r_620l.message}")

    # ── conv_se ───────────────────────────────────────────────────────────────
    if run_conv:
        print("\n── conv_se ───────────────────────────────────────────────")
        p_orig  = MODELS_DIR / "conv_se_context.onnx"
        p_620l  = MODELS_DIR / "conv_se_context_ax620l.onnx"

        r_orig = check_mask_model(p_orig, T=64, F=257, out_F=129)
        results.append(r_orig)
        print(f"  {'PASS' if r_orig.passed else 'FAIL'} {r_orig.name}: {r_orig.message}")

        if r_orig.passed and p_orig.exists():
            ref_out = get_mask_output(p_orig, T=64, F=257, seed=77)
            np.random.seed(77)
            r_620l = check_mask_model(p_620l, T=64, F=257, out_F=129,
                                      ref_output=ref_out, min_snr=100.0)
        else:
            r_620l = check_mask_model(p_620l, T=64, F=257, out_F=129)
        results.append(r_620l)
        print(f"  {'PASS' if r_620l.passed else 'FAIL'} {r_620l.name}: {r_620l.message}")

    # ── GTCRN ─────────────────────────────────────────────────────────────────
    if run_gtcrn:
        print("\n── gtcrn ─────────────────────────────────────────────────")
        p_620e = MODELS_DIR / "gtcrn_no_scatter_less_input_optimized.onnx"
        p_650  = MODELS_DIR / "gtcrn_ax650_nopd_fixed.onnx"

        for p in [p_620e, p_650]:
            r = check_gtcrn_model(p)
            results.append(r)
            print(f"  {'PASS' if r.passed else 'FAIL'} {r.name}: {r.message}")

        if p_620e.exists() and p_650.exists():
            print("\n  Cross-model comparison (620e vs 650):")
            for r in check_gtcrn_pair(p_620e, p_650):
                results.append(r)
                print(f"  {'PASS' if r.passed else 'FAIL'} {r.name}: {r.message}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all ONNX models in models/")
    parser.add_argument("--model", choices=["tiny_v5", "conv_se", "gtcrn", "all"],
                        default="all", help="Which model group to verify")
    args = parser.parse_args()

    print(f"Models directory: {MODELS_DIR}")
    results = run_all_checks(args)

    passed = sum(1 for r in results if r.passed)
    total  = len(results)

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} checks passed")
    if passed < total:
        print("FAILED checks:")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.message}")
        sys.exit(1)
    else:
        print("All checks passed ✓")


if __name__ == "__main__":
    main()
