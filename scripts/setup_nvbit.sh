#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NVBIT_VERSION="1.7.7.3"

NVBIT_DIR="$ROOT/dependencies/nvbit-$NVBIT_VERSION"
DOWNLOAD_DIR="$ROOT/dependencies/downloads"

ARCHIVE="nvbit-Linux-x86_64-$NVBIT_VERSION.tar.bz2"
ARCHIVE_PATH="$DOWNLOAD_DIR/$ARCHIVE"

URL="https://github.com/NVlabs/NVBit/releases/download/v$NVBIT_VERSION/$ARCHIVE"

echo "=== SuperCollider NVBit setup ==="
echo "ROOT=$ROOT"
echo "NVBit version=$NVBIT_VERSION"

mkdir -p "$ROOT/dependencies"
mkdir -p "$DOWNLOAD_DIR"

# Already installed.
if [ -f "$NVBIT_DIR/core/nvbit.h" ] && \
   [ -f "$NVBIT_DIR/core/libnvbit.a" ]; then
    echo "[NVBIT] NVBit $NVBIT_VERSION already installed"
    echo "[NVBIT] NVBIT_HOME=$NVBIT_DIR"
    exit 0
fi

# Download release archive.
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "[NVBIT] downloading:"
    echo "        $URL"

    wget \
        -O "$ARCHIVE_PATH" \
        "$URL"
else
    echo "[NVBIT] archive already downloaded:"
    echo "        $ARCHIVE_PATH"
fi

# Extract into a temporary location first.
TMP_DIR="$ROOT/dependencies/.nvbit-extract"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "[NVBIT] extracting..."

tar -xjf "$ARCHIVE_PATH" -C "$TMP_DIR"

EXTRACTED_DIR="$(
    find "$TMP_DIR" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -print \
        -quit
)"

if [ -z "$EXTRACTED_DIR" ]; then
    echo "ERROR: could not locate extracted NVBit directory"
    exit 1
fi

# Replace incomplete/old installation.
rm -rf "$NVBIT_DIR"
mv "$EXTRACTED_DIR" "$NVBIT_DIR"

rm -rf "$TMP_DIR"

# Validate installation.
if [ ! -f "$NVBIT_DIR/core/nvbit.h" ]; then
    echo "ERROR: core/nvbit.h not found after extraction"
    exit 1
fi

if [ ! -f "$NVBIT_DIR/core/libnvbit.a" ]; then
    echo "ERROR: core/libnvbit.a not found after extraction"
    exit 1
fi

if [ ! -d "$NVBIT_DIR/tools" ]; then
    echo "ERROR: NVBit tools directory not found"
    exit 1
fi

echo
echo "=== NVBit installation complete ==="
echo "NVBIT_HOME=$NVBIT_DIR"

echo
echo "Core:"
echo "  $NVBIT_DIR/core/nvbit.h"
echo "  $NVBIT_DIR/core/libnvbit.a"

echo
echo "Available tools:"

find "$NVBIT_DIR/tools" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -printf "  %f\n" \
    | sort
