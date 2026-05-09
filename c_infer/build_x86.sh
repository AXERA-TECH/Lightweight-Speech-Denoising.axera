#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_x86"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
  -DBUILD_TARGET=x86 \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"

echo "Build complete: ${BUILD_DIR}/test_se_denoise"
echo "Example:"
echo "  ${BUILD_DIR}/test_se_denoise ${SCRIPT_DIR}/../test_wavs/mix.wav ${SCRIPT_DIR}/output/out_tiny_v5.wav ${SCRIPT_DIR}/models/tiny_v5_context_config.ini"

