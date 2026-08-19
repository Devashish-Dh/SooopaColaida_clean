#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLVM_SRC="$ROOT/dependencies/llvm-project"
LLVM_BUILD="$ROOT/dependencies/build-llvm"
LLVM_INSTALL="$ROOT/dependencies/INSTALL-llvm"

LLVM_REPO="https://github.com/llvm/llvm-project.git"
LLVM_COMMIT="ca7933e47d3a3451d81e72ac174dcb5aa28b59d1"

echo "=== SuperCollider LLVM setup ==="
echo "ROOT=$ROOT"

mkdir -p "$ROOT/dependencies"

if [ ! -d "$LLVM_SRC/.git" ]; then
    echo "[LLVM] cloning llvm-project..."
    git clone "$LLVM_REPO" "$LLVM_SRC"
else
    echo "[LLVM] source already present"
fi

CURRENT_COMMIT="$(git -C "$LLVM_SRC" rev-parse HEAD)"

if [ "$CURRENT_COMMIT" != "$LLVM_COMMIT" ]; then
    echo "[LLVM] checking out pinned revision:"
    echo "       $LLVM_COMMIT"

    git -C "$LLVM_SRC" fetch origin
    git -C "$LLVM_SRC" checkout "$LLVM_COMMIT"
else
    echo "[LLVM] correct revision already checked out"
fi

if [ -x "$LLVM_INSTALL/bin/llvm-config" ]; then
    INSTALLED_VERSION="$("$LLVM_INSTALL/bin/llvm-config" --version)"

    if [ "$INSTALLED_VERSION" = "22.1.8" ]; then
        echo "[LLVM] LLVM 22.1.8 already installed"
        "$LLVM_INSTALL/bin/clang" --version | head -n 1
        "$LLVM_INSTALL/bin/clangd" --version | head -n 1
        exit 0
    fi
fi

echo "[LLVM] configuring..."

cmake \
    -S "$LLVM_SRC/llvm" \
    -B "$LLVM_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF

echo "[LLVM] building..."

cmake --build "$LLVM_BUILD" -j96

echo "[LLVM] installing..."

cmake --install "$LLVM_BUILD"

echo
echo "=== LLVM installation complete ==="

"$LLVM_INSTALL/bin/llvm-config" --version
"$LLVM_INSTALL/bin/clang" --version | head -n 1
"$LLVM_INSTALL/bin/clangd" --version | head -n 1
"$LLVM_INSTALL/bin/opt" --version | head -n 1
