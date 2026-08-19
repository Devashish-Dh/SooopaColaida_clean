#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_iiwlp.sh [options]

Options:
  -n, --runs N          Number of runs. Default: 50
  -b, --build-dir DIR   Directory containing the built executable.
                        Default: ./build-artifacts
  -o, --log-dir DIR     Directory where timestamped logs are written.
                        Default: ./run-logs
  --sc-runtime FILE     SuperCollider reporting runtime to LD_PRELOAD.
                        Default: ../../project/build/libsc.so
  -h, --help            Show this help.

Environment overrides:
  RUNS, BUILD_DIR, LOG_ROOT, CUDA_HOME, SC_RUNTIME, SC_RUNTIME_SO
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_NAME="intra_warp_lost_update"

prepend_path() {
    local var_name="$1"
    local new_dir="$2"
    local current_value

    if [ ! -d "$new_dir" ]; then
        return
    fi

    current_value="${!var_name:-}"
    case ":$current_value:" in
        *":$new_dir:"*)
            ;;
        *)
            if [ -n "$current_value" ]; then
                export "$var_name=$new_dir:$current_value"
            else
                export "$var_name=$new_dir"
            fi
            ;;
    esac
}

RUNS="${RUNS:-50}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build-artifacts}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/run-logs}"
CUDA_HOME="${CUDA_HOME:-/disk/devashish/CUDA_Toolkits/cuda-12.9}"
SC_RUNTIME="${SC_RUNTIME:-${SC_RUNTIME_SO:-$PROJECT_ROOT/project/build/libsc.so}}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--runs)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            RUNS="$2"
            shift 2
            ;;
        -b|--build-dir)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            BUILD_DIR="$2"
            shift 2
            ;;
        -o|--log-dir)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        --sc-runtime)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            SC_RUNTIME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$RUNS" in
    ''|*[!0-9]*)
        echo "error: runs must be a positive integer, got '$RUNS'" >&2
        exit 2
        ;;
esac

if [ "$RUNS" -lt 1 ]; then
    echo "error: runs must be at least 1" >&2
    exit 2
fi

EXE="$BUILD_DIR/$TEST_NAME"
if [ ! -x "$EXE" ]; then
    echo "error: executable does not exist or is not executable: $EXE" >&2
    echo "hint: run make in $SCRIPT_DIR first" >&2
    exit 1
fi

if [ ! -f "$SC_RUNTIME" ]; then
    echo "error: SuperCollider runtime does not exist: $SC_RUNTIME" >&2
    echo "hint: build the project first: cmake --build $PROJECT_ROOT/project/build -j" >&2
    exit 1
fi

export CUDA_HOME
prepend_path PATH "$CUDA_HOME/bin"
prepend_path LD_LIBRARY_PATH "$(dirname "$SC_RUNTIME")"
prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"
prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib"
prepend_path LD_LIBRARY_PATH "$CUDA_HOME/targets/x86_64-linux/lib"
prepend_path LD_LIBRARY_PATH "$CUDA_HOME/extras/CUPTI/lib64"

RUN_LD_PRELOAD="$SC_RUNTIME"
if [ -n "${LD_PRELOAD:-}" ]; then
    RUN_LD_PRELOAD="$RUN_LD_PRELOAD:$LD_PRELOAD"
fi

RUN_DIR="$LOG_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/$TEST_NAME.log"
SUMMARY="$RUN_DIR/summary.tsv"
ENV_LOG="$RUN_DIR/runtime-env.txt"

{
    printf 'SC_RUNTIME=%s\n' "$SC_RUNTIME"
    printf 'LD_PRELOAD=%s\n' "$RUN_LD_PRELOAD"
    printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
    printf 'PATH=%s\n' "${PATH:-}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
} > "$ENV_LOG"

printf 'test\truns\tpassed\tfailed\tlog\n' > "$SUMMARY"

{
    printf 'test: %s\n' "$TEST_NAME"
    printf 'exe: %s\n' "$EXE"
    printf 'ld_preload: %s\n' "$RUN_LD_PRELOAD"
    printf 'runs: %s\n' "$RUNS"
    printf 'started: %s\n' "$(date --iso-8601=seconds)"
    printf '\n'
} > "$LOG"

echo "run directory: $RUN_DIR"
echo "runs: $RUNS"
echo "SC runtime: $SC_RUNTIME"
echo "runtime env: $ENV_LOG"
echo "log: $LOG"

passed=0
failed=0

for ((run = 1; run <= RUNS; ++run)); do
    start_epoch="$(date +%s)"
    {
        printf '===== run %03d/%03d start %s =====\n' \
            "$run" "$RUNS" "$(date --iso-8601=seconds)"
    } >> "$LOG"

    (cd "$(dirname "$EXE")" && LD_PRELOAD="$RUN_LD_PRELOAD" "./$TEST_NAME") >> "$LOG" 2>&1
    status="$?"

    end_epoch="$(date +%s)"
    duration="$((end_epoch - start_epoch))"
    {
        printf '===== run %03d/%03d exit %d duration_s %d =====\n' \
            "$run" "$RUNS" "$status" "$duration"
        printf '\n'
    } >> "$LOG"

    if [ "$status" -eq 0 ]; then
        passed="$((passed + 1))"
    else
        failed="$((failed + 1))"
    fi
done

printf '%s\t%s\t%s\t%s\t%s\n' \
    "$TEST_NAME" "$RUNS" "$passed" "$failed" "$LOG" >> "$SUMMARY"

echo
echo "summary: $SUMMARY"
echo "passed runs: $passed"
echo "failed runs: $failed"
