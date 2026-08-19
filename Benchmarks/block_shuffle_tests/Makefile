MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SOOPA_COLAIDA_ROOT ?= $(abspath $(MAKEFILE_DIR)/../..)

PROJECT_ROOT ?= $(SOOPA_COLAIDA_ROOT)/project
LLVM_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/cuda-12.1

SC_PLUGIN ?= $(PROJECT_ROOT)/build/SuperColliderPass.so
SC_RUNTIME ?= $(PROJECT_ROOT)/build/libsc.so

# Detect GPU 0 compute capability.
# Examples:
#   7.5 -> 75
#   8.6 -> 86
DETECTED_CC := $(shell \
	nvidia-smi \
		--query-gpu=compute_cap \
		--format=csv,noheader \
		--id=0 2>/dev/null \
		| head -n 1 \
		| tr -d '.[:space:]')

ifeq ($(DETECTED_CC),)
$(error Could not detect GPU compute capability with nvidia-smi)
endif

ARCH ?= sm_$(DETECTED_CC)
COMPUTE ?= compute_$(DETECTED_CC)

CLANGXX := $(LLVM_HOME)/bin/clang++
OPT := $(LLVM_HOME)/bin/opt
LLVM_LINK := $(LLVM_HOME)/bin/llvm-link
LLC := $(LLVM_HOME)/bin/llc
PTXAS := $(CUDA_HOME)/bin/ptxas
FATBINARY := $(CUDA_HOME)/bin/fatbinary

MAIN_SRC := block_shuffle_tests.cu
PREPASS_HELPER_SRC := device_block_helpers.cu
POSTPASS_PHYSICAL_SRC := physical_identity_after_pass.cu
INLINE_REJECT_SRC := inline_ptx_reject.cu
YZ_ONLY_SRC := yz_only.cu

BUILD_ROOT := build-artifacts
VARIANT ?= seed12345
SHUFFLE ?= 1
SEED ?= 12345

VARIANT_DIR := $(BUILD_ROOT)/$(VARIANT)

MAIN_RAW_LL := $(VARIANT_DIR)/main.device.raw.ll
PREPASS_HELPER_RAW_LL := $(VARIANT_DIR)/prepass_helper.device.raw.ll
PREPASS_COMBINED_RAW_LL := $(VARIANT_DIR)/combined.device.raw.ll
O1_LL := $(VARIANT_DIR)/combined.device.o1.ll
SC_LL := $(VARIANT_DIR)/combined.device.sc.ll
PASS_LOG := $(VARIANT_DIR)/supercollider.log

POSTPASS_RAW_LL := $(VARIANT_DIR)/physical_after_pass.device.raw.ll
POSTPASS_LL := $(VARIANT_DIR)/physical_after_pass.device.o1.ll
LINKED_LL := $(VARIANT_DIR)/combined.device.linked.ll

PTX := $(VARIANT_DIR)/block_shuffle_tests.ptx
CUBIN := $(VARIANT_DIR)/block_shuffle_tests.cubin
FATBIN := $(VARIANT_DIR)/block_shuffle_tests.fatbin

HOST_RAW_LL := $(VARIANT_DIR)/block_shuffle_tests.host.raw.ll
HOST_LL := $(VARIANT_DIR)/block_shuffle_tests.host.ll
HOST_OBJ := $(VARIANT_DIR)/block_shuffle_tests.host.o
EXE := $(VARIANT_DIR)/block_shuffle_tests

PASS_PIPELINE = supercollider<rdelay=0;wdelay=0;ilu=0;async=0;bulk=0;shuffle=$(SHUFFLE);shuffle_seed=$(SEED)>

CUDA_COMMON_FLAGS := \
	--cuda-path=$(CUDA_HOME) \
	--cuda-gpu-arch=$(ARCH) \
	-I$(CUDA_HOME)/include \
	-std=c++17 \
	-gline-tables-only \
	-O0 \
	-Xclang -disable-llvm-passes \
	-Xclang -disable-O0-optnone \
	-fno-discard-value-names

DEVICE_FLAGS := $(CUDA_COMMON_FLAGS) -fgpu-rdc
HOST_FLAGS := $(CUDA_COMMON_FLAGS)

.PHONY: \
	all \
	build-one \
	baseline \
	seed12345 \
	seed987654321 \
	run \
	check \
	inspect \
	check-static \
	check-runtime \
	check-inline-reject \
	check-yz-only \
	info \
	clean \
	help

all: baseline seed12345 seed987654321 check-inline-reject check-yz-only

info:
	@echo "SOOPA_COLAIDA_ROOT = $(SOOPA_COLAIDA_ROOT)"
	@echo "PROJECT_ROOT       = $(PROJECT_ROOT)"
	@echo "LLVM_HOME          = $(LLVM_HOME)"
	@echo "CUDA_HOME          = $(CUDA_HOME)"
	@echo "SC_PLUGIN          = $(SC_PLUGIN)"
	@echo "SC_RUNTIME         = $(SC_RUNTIME)"
	@echo "Detected CC        = $(DETECTED_CC)"
	@echo "ARCH               = $(ARCH)"
	@echo "COMPUTE            = $(COMPUTE)"

baseline:
	@$(MAKE) --no-print-directory \
		build-one \
		VARIANT=baseline \
		SHUFFLE=0 \
		SEED=0

seed12345:
	@$(MAKE) --no-print-directory \
		build-one \
		VARIANT=seed12345 \
		SHUFFLE=1 \
		SEED=12345

seed987654321:
	@$(MAKE) --no-print-directory \
		build-one \
		VARIANT=seed987654321 \
		SHUFFLE=1 \
		SEED=987654321

build-one: $(EXE)

$(VARIANT_DIR):
	@mkdir -p $@

$(MAIN_RAW_LL): $(MAIN_SRC) | $(VARIANT_DIR)
	$(CLANGXX) \
		$(DEVICE_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  DEVICE RAW      -> $@"

$(PREPASS_HELPER_RAW_LL): $(PREPASS_HELPER_SRC) | $(VARIANT_DIR)
	$(CLANGXX) \
		$(DEVICE_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  PREPASS HELPER  -> $@"

$(PREPASS_COMBINED_RAW_LL): $(MAIN_RAW_LL) $(PREPASS_HELPER_RAW_LL)
	$(LLVM_LINK) \
		-S \
		$(MAIN_RAW_LL) \
		$(PREPASS_HELPER_RAW_LL) \
		-o $@
	@echo "  PREPASS LINK    -> $@"

$(O1_LL): $(PREPASS_COMBINED_RAW_LL)
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@
	@echo "  DEVICE O1       -> $@"

$(SC_LL): $(O1_LL)
	@set -e; \
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(PASS_PIPELINE)' \
		-S $(O1_LL) \
		-o $(SC_LL) \
		2>$(PASS_LOG); \
	echo "  SUPERCOLLIDER   -> $(SC_LL)"; \
	if [ "$(SHUFFLE)" = "1" ]; then \
		grep -A6 '\[SuperCollider\] Block shuffling' $(PASS_LOG) || true; \
	fi

$(POSTPASS_RAW_LL): $(POSTPASS_PHYSICAL_SRC) | $(VARIANT_DIR)
	$(CLANGXX) \
		$(DEVICE_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  POSTPASS RAW    -> $@"

$(POSTPASS_LL): $(POSTPASS_RAW_LL)
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@
	@echo "  POSTPASS O1     -> $@"

$(LINKED_LL): $(SC_LL) $(POSTPASS_LL)
	$(LLVM_LINK) \
		-S \
		$(SC_LL) \
		$(POSTPASS_LL) \
		-o $@
	@echo "  FINAL IR LINK   -> $@"

$(PTX): $(LINKED_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(ARCH) \
		$< \
		-o $@
	@echo "  PTX             -> $@"

$(CUBIN): $(PTX)
	$(PTXAS) \
		--gpu-name=$(ARCH) \
		$< \
		-o $@
	@echo "  CUBIN           -> $@"

$(FATBIN): $(CUBIN) $(PTX)
	$(FATBINARY) \
		--create=$@ \
		--64 \
		--image=profile=$(ARCH),file=$(CUBIN) \
		--image=profile=$(COMPUTE),file=$(PTX)
	@echo "  FATBIN          -> $@"

$(HOST_RAW_LL): $(MAIN_SRC) $(FATBIN) | $(VARIANT_DIR)
	$(CLANGXX) \
		$(HOST_FLAGS) \
		-DSC_EXPECT_SHUFFLE=$(SHUFFLE) \
		-DSC_TEST_SHUFFLE_SEED=$(SEED)ULL \
		--cuda-host-only \
		-Xclang -fcuda-include-gpubinary \
		-Xclang $(abspath $(FATBIN)) \
		-S \
		-emit-llvm \
		$(MAIN_SRC) \
		-o $@
	@echo "  HOST RAW        -> $@"

$(HOST_LL): $(HOST_RAW_LL)
	$(OPT) \
		-passes=mem2reg \
		-S \
		$< \
		-o $@
	@echo "  HOST LLVM       -> $@"

$(HOST_OBJ): $(HOST_LL)
	$(CLANGXX) \
		-c \
		$< \
		-o $@
	@echo "  HOST OBJ        -> $@"

$(EXE): $(HOST_OBJ)
	$(CLANGXX) \
		$< \
		-L$(CUDA_HOME)/lib64 \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-lcudart \
		-o $@
	@echo "  EXE             -> $@"

run: all
	@echo
	@echo "=== baseline: shuffle=0 ==="
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/baseline/block_shuffle_tests
	@echo
	@echo "=== shuffle=1 seed=12345 ==="
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/seed12345/block_shuffle_tests
	@echo
	@echo "=== shuffle=1 seed=987654321 ==="
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/seed987654321/block_shuffle_tests

check-static: all
	@echo "=== Static IR/PTX checks ==="
	@! grep -q 'sc\.shuffle\.logical\.x' \
		$(BUILD_ROOT)/baseline/combined.device.sc.ll || \
		{ echo "ERROR: shuffle=0 unexpectedly generated block-shuffle IR"; exit 1; }

	@grep -q 'sc\.shuffle\.logical\.x' \
		$(BUILD_ROOT)/seed12345/combined.device.sc.ll || \
		{ echo "ERROR: shuffle=1 did not generate logical block IR"; exit 1; }

	@grep -q 'llvm\.nvvm\.read\.ptx\.sreg\.gridid' \
		$(BUILD_ROOT)/seed12345/combined.device.sc.ll || \
		{ echo "ERROR: shuffled IR does not read gridid"; exit 1; }

	@grep -q 'sc_test_physical_block_x' \
		$(BUILD_ROOT)/seed12345/combined.device.linked.ll || \
		{ echo "ERROR: post-pass physical-ID helper missing"; exit 1; }

	@grep -q '%gridid' \
		$(BUILD_ROOT)/seed12345/block_shuffle_tests.ptx || \
		{ echo "ERROR: final PTX does not reference %gridid"; exit 1; }

	@grep -Eq '^[[:space:]]*\.visible[[:space:]]+\.entry[[:space:]]+mappingKernel\(' \
		$(BUILD_ROOT)/seed12345/block_shuffle_tests.ptx || \
		{ echo "ERROR: mappingKernel is not a visible PTX entry; host/device kernel naming may disagree"; exit 1; }

	@grep -Eq '^[[:space:]]*\.visible[[:space:]]+\.entry[[:space:]]+threeDimensionalKernel\(' \
		$(BUILD_ROOT)/seed12345/block_shuffle_tests.ptx || \
		{ echo "ERROR: threeDimensionalKernel is not a visible PTX entry"; exit 1; }

	@echo "PASS: static shuffled/unshuffled IR and PTX checks"

check-runtime: all
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/baseline/block_shuffle_tests
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/seed12345/block_shuffle_tests
	@LD_PRELOAD=$(SC_RUNTIME) ./$(BUILD_ROOT)/seed987654321/block_shuffle_tests
	@echo "PASS: all runtime variants"

check-inline-reject: $(BUILD_ROOT)/negative/inline_reject.log
	@grep -q 'handwritten inline PTX reads %ctaid.x' $< || \
		{ echo "ERROR: inline-PTX rejection message not found"; cat $<; exit 1; }
	@echo "PASS: handwritten inline PTX %ctaid.x is rejected"

$(BUILD_ROOT)/negative/inline_reject.raw.ll: $(INLINE_REJECT_SRC)
	@mkdir -p $(BUILD_ROOT)/negative
	$(CLANGXX) \
		$(DEVICE_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@

$(BUILD_ROOT)/negative/inline_reject.o1.ll: $(BUILD_ROOT)/negative/inline_reject.raw.ll
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@

$(BUILD_ROOT)/negative/inline_reject.log: $(BUILD_ROOT)/negative/inline_reject.o1.ll
	@set +e; \
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='supercollider<rdelay=0;wdelay=0;ilu=0;async=0;bulk=0;shuffle=1;shuffle_seed=12345>' \
		-S $< \
		-o $(BUILD_ROOT)/negative/inline_reject.sc.ll \
		2>$@; \
	status=$$?; \
	set -e; \
	if [ $$status -eq 0 ]; then \
		echo "ERROR: inline PTX %ctaid.x module was unexpectedly accepted"; \
		exit 1; \
	fi

check-yz-only: $(BUILD_ROOT)/yz-only/yz_only.log
	@grep -A6 '\[SuperCollider\] Block shuffling' $< | \
		grep -Eq 'blockIdx\.x reads[[:space:]]*:[[:space:]]*0' || \
		{ echo "ERROR: y/z-only module unexpectedly contained x candidates"; cat $<; exit 1; }

	@grep -A6 '\[SuperCollider\] Block shuffling' $< | \
		grep -Eq 'rewritten[[:space:]]*:[[:space:]]*0' || \
		{ echo "ERROR: y/z-only module unexpectedly rewrote blockIdx.x"; cat $<; exit 1; }

	@echo "PASS: blockIdx.y/z-only module has zero x rewrites"

$(BUILD_ROOT)/yz-only/yz_only.raw.ll: $(YZ_ONLY_SRC)
	@mkdir -p $(BUILD_ROOT)/yz-only
	$(CLANGXX) \
		$(DEVICE_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@

$(BUILD_ROOT)/yz-only/yz_only.o1.ll: $(BUILD_ROOT)/yz-only/yz_only.raw.ll
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@

$(BUILD_ROOT)/yz-only/yz_only.log: $(BUILD_ROOT)/yz-only/yz_only.o1.ll
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='supercollider<rdelay=0;wdelay=0;ilu=0;async=0;bulk=0;shuffle=1;shuffle_seed=12345>' \
		-S $< \
		-o $(BUILD_ROOT)/yz-only/yz_only.sc.ll \
		2>$@

check: check-static check-inline-reject check-yz-only check-runtime
	@echo
	@echo "ALL BLOCK-SHUFFLE TESTS PASSED"

inspect: all
	@echo "=== shuffle=1 pass summary (seed 12345) ==="
	@grep -A6 '\[SuperCollider\] Block shuffling' \
		$(BUILD_ROOT)/seed12345/supercollider.log || true

	@echo
	@echo "=== generated shuffle IR ==="
	@grep -nE 'sc\.shuffle\.(physical|grid|gridid|a|b|logical)' \
		$(BUILD_ROOT)/seed12345/combined.device.sc.ll | head -80 || true

	@echo
	@echo "=== final PTX special-register reads ==="
	@grep -nE '%(ctaid\.x|nctaid\.x|gridid)' \
		$(BUILD_ROOT)/seed12345/block_shuffle_tests.ptx | head -80 || true

	@echo
	@echo "=== negative inline-PTX guard ==="
	@grep -E 'block shuffling|inline PTX|ctaid\.x' \
		$(BUILD_ROOT)/negative/inline_reject.log | tail -20 || true

	@echo
	@echo "=== y/z-only summary ==="
	@grep -A6 '\[SuperCollider\] Block shuffling' \
		$(BUILD_ROOT)/yz-only/yz_only.log || true

help:
	@echo "make                 Build baseline + two shuffled seeds and compile-only guards"
	@echo "make info            Show detected paths and GPU architecture"
	@echo "make inspect         Show concise pass/IR/PTX evidence"
	@echo "make check           Run static, negative, y/z-only, and runtime tests"
	@echo "make run             Run baseline and both shuffled runtime suites"
	@echo "make clean           Remove build-artifacts"
	@echo
	@echo "Overrides:"
	@echo "  ARCH=sm_86 COMPUTE=compute_86"
	@echo "  LLVM_HOME=..."
	@echo "  CUDA_HOME=..."
	@echo "  PROJECT_ROOT=..."

clean:
	rm -rf $(BUILD_ROOT)