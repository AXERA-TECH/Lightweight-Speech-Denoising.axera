#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ax620q"

# 根据实际路径修改以下两项
BSP_SDK_DIR="/home/hy/audio_project/third_party/ax620e_bsp_sdk"           # AX620Q/E BSP SDK 路径
TOOLCHAIN_DIR="/home/hy/audio_project/third_party/arm-AX620E-linux-uclibcgnueabihf"  # ARM32 uclibc 工具链路径

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain_ax620q.cmake" \
  -DTOOLCHAIN_DIR="${TOOLCHAIN_DIR}" \
  -DBUILD_TARGET=ax620q \
  -DBSP_SDK_DIR="${BSP_SDK_DIR}" \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"
