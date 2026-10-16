#!/bin/sh
# multimodal_model_demo native build script - adapted for the Debian @PACKAGE@ package.
# Upstream original pointed at a cross toolchain under ~/opts/... which does not
# exist on the board. This version builds natively with the system gcc.
set -e

BUILD_TYPE="${BUILD_TYPE:-Release}"

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${ROOT_PWD}/build/build_linux_${BUILD_TYPE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ../.. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make -j"$(nproc 2>/dev/null || echo 4)"
make install

echo "Built binaries in: ${BUILD_DIR}/install/"
