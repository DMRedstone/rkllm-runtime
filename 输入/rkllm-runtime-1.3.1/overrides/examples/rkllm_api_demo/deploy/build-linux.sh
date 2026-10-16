#!/bin/sh
# rkllm_api_demo native build script - adapted for the Debian @PACKAGE@ package.
# Upstream original required a cross toolchain at
# ~/opts/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu, which does not exist
# on the board itself. It also used an undefined $SOURCE variable in the path
# resolution. Both are fixed below: this builds natively with the system gcc.
set -e

BUILD_TYPE="${BUILD_TYPE:-Release}"

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${ROOT_PWD}/build/build_linux_${BUILD_TYPE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ../.. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make -j"$(nproc 2>/dev/null || echo 4)"

echo "Built: ${BUILD_DIR}/llm_demo"
