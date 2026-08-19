#!/usr/bin/env bash

SOOPA_COLAIDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------
# Repo-local dependencies
# ------------------------------------------------------------

export LLVM_HOME="$SOOPA_COLAIDA_ROOT/dependencies/INSTALL-llvm"
export CUDA_HOME="$SOOPA_COLAIDA_ROOT/dependencies/cuda-12.1"
export NVBIT_HOME="$SOOPA_COLAIDA_ROOT/dependencies/nvbit-1.7.7.3"

# ------------------------------------------------------------
# LLVM
# ------------------------------------------------------------

export LLVM_DIR="$LLVM_HOME/lib/cmake/llvm"
export CLANG_DIR="$LLVM_HOME/lib/cmake/clang"
export LLVM_CONFIG="$LLVM_HOME/bin/llvm-config"

export CLANG="$LLVM_HOME/bin/clang"
export CLANGXX="$LLVM_HOME/bin/clang++"
export CLANGD="$LLVM_HOME/bin/clangd"

export OPT="$LLVM_HOME/bin/opt"
export LLC="$LLVM_HOME/bin/llc"
export LLVM_LINK="$LLVM_HOME/bin/llvm-link"

# ------------------------------------------------------------
# Environment paths
# ------------------------------------------------------------

export PATH="$LLVM_HOME/bin:$CUDA_HOME/bin:$PATH"

export LD_LIBRARY_PATH="$LLVM_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CMAKE_PREFIX_PATH="$LLVM_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

echo "============================================================"
echo " SooopaColaida environment"
echo "============================================================"
echo "ROOT       = $SOOPA_COLAIDA_ROOT"
echo "LLVM_HOME  = $LLVM_HOME"
echo "CUDA_HOME  = $CUDA_HOME"
echo "NVBIT_HOME = $NVBIT_HOME"
echo

if [ ! -x "$LLVM_CONFIG" ]; then
    echo "[ERROR] LLVM not found:"
    echo "        $LLVM_CONFIG"
    return 1 2>/dev/null || exit 1
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[ERROR] CUDA toolkit not found:"
    echo "        $CUDA_HOME"
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$NVBIT_HOME/core/nvbit.h" ]; then
    echo "[ERROR] NVBit not found:"
    echo "        $NVBIT_HOME"
    return 1 2>/dev/null || exit 1
fi

echo "[LLVM]"
echo "  $("$LLVM_CONFIG" --version)"

echo "[clang]"
echo "  $("$CLANG" --version | head -n 1)"

echo "[clangd]"
echo "  $("$CLANGD" --version | head -n 1)"

echo "[CUDA]"
"$CUDA_HOME/bin/nvcc" --version | grep -E "release|Build"

echo "[NVBit]"
echo "  1.7.7.3"

echo
echo "[OK] environment activated"
echo "============================================================"