#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_block_shuffle_tests.sh [options]

Options:
  -n, --runs N          Runs per variant. Default: 50
  -o, --log-dir DIR     Directory where timestamped logs are written.
                        Default: ./run-logs
  -b, --build-root DIR  Directory containing variant build artifacts.
                        Default: ./build-artifacts
  --sc-runtime FILE     SuperCollider reporting runtime to LD_PRELOAD.
                        Default: ../../project/build/libsc.so
  --cuda-home DIR       CUDA toolkit root. Default: /disk/devashish/CUDA_Toolkits/cuda-12.9
  --no-build            Do not run make before executing variants.
  --static              Run make check-static before runtime execution.
  --static-only         Run make check-static and stop before runtime execution.
  --variant NAME        Run only one variant. May be passed multiple times.
                        Defaults: baseline, seed12345, seed987654321
  -h, --help            Show this help.

Environment overrides:
  RUNS, BUILD_ROOT, LOG_ROOT, CUDA_HOME, SC_RUNTIME, SC_RUNTIME_SO
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOOPA_COLAIDA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/build-artifacts}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/run-logs}"
CUDA_HOME="${CUDA_HOME:-/disk/devashish/CUDA_Toolkits/cuda-12.9}"
SC_RUNTIME="${SC_RUNTIME:-${SC_RUNTIME_SO:-$SOOPA_COLAIDA_ROOT/project/build/libsc.so}}"
DO_BUILD=1
DO_STATIC=0
STATIC_ONLY=0
variants=()

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
        -o|--log-dir)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        -b|--build-root)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            BUILD_ROOT="$2"
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
        --cuda-home)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            CUDA_HOME="$2"
            shift 2
            ;;
        --no-build)
            DO_BUILD=0
            shift
            ;;
        --static)
            DO_STATIC=1
            shift
            ;;
        --static-only)
            DO_STATIC=1
            STATIC_ONLY=1
            shift
            ;;
        --variant)
            if [ "$#" -lt 2 ]; then
                echo "error: missing value for $1" >&2
                exit 2
            fi
            variants+=("$2")
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

if [ "${#variants[@]}" -eq 0 ]; then
    variants=(baseline seed12345 seed987654321)
fi

if [ ! -f "$SC_RUNTIME" ]; then
    echo "error: SuperCollider runtime does not exist: $SC_RUNTIME" >&2
    echo "hint: build the project first: cmake --build $SOOPA_COLAIDA_ROOT/project/build -j" >&2
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

if [ "$DO_BUILD" -eq 1 ]; then
    make -C "$SCRIPT_DIR" all
fi

if [ "$DO_STATIC" -eq 1 ]; then
    make -C "$SCRIPT_DIR" check-static
fi

if [ "$STATIC_ONLY" -eq 1 ]; then
    echo "static checks passed; skipping runtime execution"
    exit 0
fi

RUN_DIR="$LOG_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

SUMMARY="$RUN_DIR/summary.tsv"
ENV_LOG="$RUN_DIR/runtime-env.txt"

printf 'variant\truns\tpassed\tfailed\tlog\n' > "$SUMMARY"

{
    printf 'SC_RUNTIME=%s\n' "$SC_RUNTIME"
    printf 'LD_PRELOAD=%s\n' "$RUN_LD_PRELOAD"
    printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
    printf 'BUILD_ROOT=%s\n' "$BUILD_ROOT"
    printf 'VARIANTS=%s\n' "${variants[*]}"
    printf 'PATH=%s\n' "${PATH:-}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
} > "$ENV_LOG"

echo "run directory: $RUN_DIR"
echo "runs per variant: $RUNS"
echo "variants: ${variants[*]}"
echo "SC runtime: $SC_RUNTIME"
echo "runtime env: $ENV_LOG"

total_passed=0
total_failed=0

for variant in "${variants[@]}"; do
    exe="$BUILD_ROOT/$variant/block_shuffle_tests"
    log="$RUN_DIR/$variant.log"
    passed=0
    failed=0

    if [ ! -x "$exe" ]; then
        echo "error: executable does not exist or is not executable: $exe" >&2
        echo "hint: run make -C $SCRIPT_DIR all" >&2
        exit 1
    fi

    {
        printf 'variant: %s\n' "$variant"
        printf 'exe: %s\n' "$exe"
        printf 'ld_preload: %s\n' "$RUN_LD_PRELOAD"
        printf 'runs: %s\n' "$RUNS"
        printf 'started: %s\n' "$(date --iso-8601=seconds)"
        printf '\n'
    } > "$log"

    echo "running $variant"

    for ((run = 1; run <= RUNS; ++run)); do
        start_epoch="$(date +%s)"
        {
            printf '===== run %03d/%03d start %s =====\n' \
                "$run" "$RUNS" "$(date --iso-8601=seconds)"
        } >> "$log"

        (cd "$(dirname "$exe")" && LD_PRELOAD="$RUN_LD_PRELOAD" ./block_shuffle_tests) >> "$log" 2>&1
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
        "$variant" "$RUNS" "$passed" "$failed" "$log" >> "$SUMMARY"

    echo "  $variant: passed=$passed failed=$failed log=$log"
done

echo
echo "summary: $SUMMARY"
echo "total passed runs: $total_passed"
echo "total failed runs: $total_failed"

if [ "$total_failed" -ne 0 ]; then
    exit 1
fi
