#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_races.sh [options]

Options:
  -n, --runs N          Runs per binary. Default: 50
  -b, --build-dir DIR   Directory containing per-test build artifacts.
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

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory does not exist: $BUILD_DIR" >&2
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

SUMMARY="$RUN_DIR/summary.tsv"
printf 'test\truns\tpassed\tfailed\tlog\n' > "$SUMMARY"

ENV_LOG="$RUN_DIR/runtime-env.txt"
{
    printf 'SC_RUNTIME=%s\n' "$SC_RUNTIME"
    printf 'LD_PRELOAD=%s\n' "$RUN_LD_PRELOAD"
    printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
    printf 'PATH=%s\n' "${PATH:-}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
} > "$ENV_LOG"

bins=()
while IFS= read -r -d '' test_dir; do
    test_name="$(basename "$test_dir")"
    exe="$test_dir/$test_name"
    if [ -x "$exe" ] && [ -f "$exe" ]; then
        bins+=("$exe")
    fi
done < <(find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [ "${#bins[@]}" -eq 0 ]; then
    echo "error: no test executables found under $BUILD_DIR" >&2
    exit 1
fi

echo "run directory: $RUN_DIR"
echo "runs per binary: $RUNS"
echo "tests found: ${#bins[@]}"
echo "SC runtime: $SC_RUNTIME"
echo "runtime env: $ENV_LOG"

total_passed=0
total_failed=0

for exe in "${bins[@]}"; do
    test_name="$(basename "$exe")"
    exe_dir="$(dirname "$exe")"
    log="$RUN_DIR/$test_name.log"
    passed=0
    failed=0

    {
        printf 'test: %s\n' "$test_name"
        printf 'exe: %s\n' "$exe"
        printf 'ld_preload: %s\n' "$RUN_LD_PRELOAD"
        printf 'runs: %s\n' "$RUNS"
        printf 'started: %s\n' "$(date --iso-8601=seconds)"
        printf '\n'
    } > "$log"

    echo "running $test_name"

    for ((run = 1; run <= RUNS; ++run)); do
        start_epoch="$(date +%s)"
        {
            printf '===== run %03d/%03d start %s =====\n' \
                "$run" "$RUNS" "$(date --iso-8601=seconds)"
        } >> "$log"

        (cd "$exe_dir" && LD_PRELOAD="$RUN_LD_PRELOAD" "./$test_name") >> "$log" 2>&1
        status="$?"

        end_epoch="$(date +%s)"
        duration="$((end_epoch - start_epoch))"
        {
            printf '===== run %03d/%03d exit %d duration_s %d =====\n' \
                "$run" "$RUNS" "$status" "$duration"
            printf '\n'
        } >> "$log"

        if [ "$status" -eq 0 ]; then
            passed="$((passed + 1))"
        else
            failed="$((failed + 1))"
        fi
    done

    total_passed="$((total_passed + passed))"
    total_failed="$((total_failed + failed))"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$test_name" "$RUNS" "$passed" "$failed" "$log" >> "$SUMMARY"
done

echo
echo "summary: $SUMMARY"
echo "passed runs: $total_passed"
echo "failed runs: $total_failed"
