#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/build_x86/test_cache_denoise"
INPUT_WAV="${1:-${SCRIPT_DIR}/../test_wavs/mix.wav}"
OUT_WAV="${2:-${SCRIPT_DIR}/output/x86_gtcrn/out_gtcrn_7input_c_onnx.wav}"

mkdir -p "$(dirname "${OUT_WAV}")"

"${BIN}" "${INPUT_WAV}" "${OUT_WAV}" "${SCRIPT_DIR}/models/gtcrn_7input_config.ini"

echo "done: ${OUT_WAV}"
