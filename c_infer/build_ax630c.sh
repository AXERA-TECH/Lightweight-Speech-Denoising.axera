#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_ax630c"

# 根据实际路径修改以下两项
BSP_SDK_DIR="/home/hy/audio_project/third_party/ax620e_bsp_sdk"           # AX630C BSP SDK 路径
TOOLCHAIN_DIR="/home/hy/audio_project/third_party/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"  # AArch64 工具链路径

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain_ax650.cmake" \
  -DTOOLCHAIN_DIR="${TOOLCHAIN_DIR}" \
  -DBUILD_TARGET=ax630c \
  -DBSP_SDK_DIR="${BSP_SDK_DIR}" \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"
