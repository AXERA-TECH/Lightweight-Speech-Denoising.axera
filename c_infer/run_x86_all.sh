#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/build_x86/test_cache_denoise"
INPUT_WAV="${1:-${SCRIPT_DIR}/../test_wavs/mix.wav}"
OUT_DIR="${2:-${SCRIPT_DIR}/output/x86_all}"

mkdir -p "${OUT_DIR}"

"${BIN}" "${INPUT_WAV}" "${OUT_DIR}/out_tiny_v5_context_c_onnx.wav" "${SCRIPT_DIR}/models/tiny_v5_context_config.ini"
"${BIN}" "${INPUT_WAV}" "${OUT_DIR}/out_conv_se_context_c_onnx.wav" "${SCRIPT_DIR}/models/conv_se_context_config.ini"
"${BIN}" "${INPUT_WAV}" "${OUT_DIR}/out_gtcrn_7input_c_onnx.wav" "${SCRIPT_DIR}/models/gtcrn_7input_config.ini"

echo "done: ${OUT_DIR}"
