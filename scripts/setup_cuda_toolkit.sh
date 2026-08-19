#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_VERSION="12.1.1"
CUDA_DIR_NAME="cuda-12.1"
CUDA_ARCH="${SC_CUDA_ARCH:-sm_75}"

AXEL_CONNECTIONS="${AXEL_CONNECTIONS:-4}"

CUDA_INSTALLER="cuda_12.1.1_530.30.02_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/$CUDA_INSTALLER"

# Expected size reported by NVIDIA for this exact runfile.
CUDA_EXPECTED_SIZE="4317456991"

DOWNLOAD_DIR="$ROOT/dependencies/downloads"
INSTALLER_PATH="$DOWNLOAD_DIR/$CUDA_INSTALLER"
CUDA_HOME="$ROOT/dependencies/$CUDA_DIR_NAME"

SMOKE_DIR="$ROOT/build/cuda-smoke"
SMOKE_SRC="$SMOKE_DIR/cuda_smoke.cu"
SMOKE_BIN="$SMOKE_DIR/cuda_smoke"

echo "=========================================="
echo " SuperCollider CUDA Toolkit setup"
echo "=========================================="
echo
echo "ROOT=$ROOT"
echo "CUDA version=$CUDA_VERSION"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_ARCH=$CUDA_ARCH"
echo "Axel connections=$AXEL_CONNECTIONS"
echo

mkdir -p "$ROOT/dependencies"
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$ROOT/build"

# ------------------------------------------------------------
# Ensure axel exists
# ------------------------------------------------------------

if ! command -v axel >/dev/null 2>&1; then
    echo "[CUDA] axel not found."

    if [ -x "$ROOT/scripts/setup_host_deps.sh" ]; then
        echo "[CUDA] running setup_host_deps.sh..."
        "$ROOT/scripts/setup_host_deps.sh"
    fi
fi

if ! command -v axel >/dev/null 2>&1; then
    echo "ERROR: axel is required but was not found."
    exit 1
fi

# ------------------------------------------------------------
# Check existing CUDA installation
# ------------------------------------------------------------

if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[CUDA] existing toolkit found"

    NVCC_OUTPUT="$("$CUDA_HOME/bin/nvcc" --version)"

    echo "$NVCC_OUTPUT"

    if echo "$NVCC_OUTPUT" | grep -q "release 12.1"; then
        echo "[CUDA] existing CUDA 12.1 installation looks valid"
    else
        echo "ERROR: unexpected CUDA installation under:"
        echo "       $CUDA_HOME"
        exit 1
    fi
else

    # --------------------------------------------------------
    # Download exact CUDA installer
    # --------------------------------------------------------

    NEED_DOWNLOAD=1

    if [ -f "$INSTALLER_PATH" ]; then
        CURRENT_SIZE="$(stat -c '%s' "$INSTALLER_PATH")"

        echo "[CUDA] existing installer size:"
        echo "       $CURRENT_SIZE bytes"

        if [ "$CURRENT_SIZE" = "$CUDA_EXPECTED_SIZE" ]; then
            echo "[CUDA] installer already completely downloaded"
            NEED_DOWNLOAD=0
        else
            echo "[CUDA] installer is incomplete"
            echo "[CUDA] expected: $CUDA_EXPECTED_SIZE bytes"
            echo "[CUDA] current:  $CURRENT_SIZE bytes"
            echo "[CUDA] attempting Axel resume..."
        fi
    fi

    if [ "$NEED_DOWNLOAD" -eq 1 ]; then
        echo
        echo "[CUDA] downloading CUDA $CUDA_VERSION"
        echo "[CUDA] using $AXEL_CONNECTIONS Axel connections"
        echo "[CUDA] $CUDA_URL"
        echo

        axel \
            -n "$AXEL_CONNECTIONS" \
            -o "$INSTALLER_PATH" \
            "$CUDA_URL"
    fi

    # --------------------------------------------------------
    # Verify complete download
    # --------------------------------------------------------

    if [ ! -f "$INSTALLER_PATH" ]; then
        echo "ERROR: CUDA installer was not downloaded"
        exit 1
    fi

    FINAL_SIZE="$(stat -c '%s' "$INSTALLER_PATH")"

    if [ "$FINAL_SIZE" != "$CUDA_EXPECTED_SIZE" ]; then
        echo "ERROR: CUDA installer size is incorrect"
        echo "Expected: $CUDA_EXPECTED_SIZE"
        echo "Actual:   $FINAL_SIZE"
        exit 1
    fi

    echo "[CUDA] installer size check passed"

    echo "[CUDA] SHA256:"
    sha256sum "$INSTALLER_PATH"

    # --------------------------------------------------------
    # Install toolkit only
    # --------------------------------------------------------

    echo
    echo "[CUDA] installing toolkit into:"
    echo "       $CUDA_HOME"
    echo

    mkdir -p "$CUDA_HOME"

    sh "$INSTALLER_PATH" \
        --silent \
        --toolkit \
        --toolkitpath="$CUDA_HOME" \
        --no-man-page
fi

# ------------------------------------------------------------
# Validate installed toolkit
# ------------------------------------------------------------

echo
echo "=== Validating CUDA installation ==="

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found"
    exit 1
fi

if [ ! -f "$CUDA_HOME/include/cuda.h" ]; then
    echo "ERROR: cuda.h not found"
    exit 1
fi

if [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
    echo "ERROR: cuda_runtime.h not found"
    exit 1
fi

if [ ! -f "$CUDA_HOME/extras/CUPTI/include/cupti.h" ]; then
    echo "ERROR: CUPTI header not found"
    exit 1
fi

if ! find "$CUDA_HOME/extras/CUPTI/lib64" \
    -maxdepth 1 \
    -name 'libcupti.so*' \
    -print \
    -quit | grep -q .; then
    echo "ERROR: libcupti.so not found"
    exit 1
fi

echo "[CUDA] nvcc:         OK"
echo "[CUDA] cuda.h:       OK"
echo "[CUDA] cuda_runtime: OK"
echo "[CUDA] CUPTI header: OK"
echo "[CUDA] CUPTI library: OK"

"$CUDA_HOME/bin/nvcc" --version

# ------------------------------------------------------------
# Build CUDA smoke test
# ------------------------------------------------------------

echo
echo "=== Building CUDA smoke test ==="

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

cat > "$SMOKE_SRC" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void smoke_kernel(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
}

int main() {
    int *device_value = nullptr;
    int host_value = 0;

    cudaError_t err = cudaMalloc(&device_value, sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMalloc failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    smoke_kernel<<<1, 1>>>(device_value);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(device_value);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(device_value);
        return 1;
    }

    err = cudaMemcpy(
        &host_value,
        device_value,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaFree(device_value);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMemcpy failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    if (host_value != 42) {
        std::fprintf(stderr,
                     "unexpected result: %d\n",
                     host_value);
        return 1;
    }

    std::printf("CUDA smoke test passed: %d\n", host_value);
    return 0;
}
EOF

"$CUDA_HOME/bin/nvcc" \
    -arch="$CUDA_ARCH" \
    "$SMOKE_SRC" \
    -o "$SMOKE_BIN"

echo "[CUDA] smoke test compiled successfully"

# ------------------------------------------------------------
# Run CUDA smoke test
# ------------------------------------------------------------

echo
echo "=== Running CUDA smoke test ==="

"$SMOKE_BIN"

echo
echo "=========================================="
echo " CUDA Toolkit setup complete"
echo "=========================================="
echo
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_ARCH=$CUDA_ARCH"
echo
echo "Add/source the following environment:"
echo
echo "  export CUDA_HOME=\"$CUDA_HOME\""
echo '  export PATH="$CUDA_HOME/bin:$PATH"'
echo '  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"'
