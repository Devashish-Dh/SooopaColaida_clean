MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SOOPA_COLAIDA_ROOT ?= $(abspath $(MAKEFILE_DIR)/../..)

LLVM_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/cuda-12.1
SC_PLUGIN ?= $(SOOPA_COLAIDA_ROOT)/project/build/SuperColliderPass.so

# Detect GPU 0 compute capability.
# Example:
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

# Default to the current GPU.
# Can be overridden from the command line.
CLASSIC_ARCH ?= sm_$(DETECTED_CC)
CLASSIC_COMPUTE ?= compute_$(DETECTED_CC)

# cp.async.bulk is a Hopper feature, so keep this target explicit.
BULK_ARCH ?= sm_90

SC_RDELAY_NS ?= 200
SC_WDELAY_NS ?= 200
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

SC_SHUFFLE_SEED_ARG := $(if $(SC_SHUFFLE_SEED),;shuffle_seed=$(SC_SHUFFLE_SEED),)

SC_PASS := supercollider<rdelay=$(SC_RDELAY_NS);wdelay=$(SC_WDELAY_NS);ilu=$(SC_ILU);async=$(SC_ASYNC);bulk=$(SC_BULK);shuffle=$(SC_SHUFFLE)$(SC_SHUFFLE_SEED_ARG)>

BULK_SC_PASS := supercollider<rdelay=$(SC_RDELAY_NS);wdelay=$(SC_WDELAY_NS);ilu=$(SC_ILU);async=$(SC_ASYNC);bulk=1;shuffle=$(SC_SHUFFLE)$(SC_SHUFFLE_SEED_ARG)>

CLANGXX := $(LLVM_HOME)/bin/clang++
OPT := $(LLVM_HOME)/bin/opt
LLC := $(LLVM_HOME)/bin/llc
PTXAS := $(CUDA_HOME)/bin/ptxas
FATBINARY := $(CUDA_HOME)/bin/fatbinary

SRC := async_ops.cu
BUILD_DIR := build-artifacts

CLASSIC_DIR := $(BUILD_DIR)/classic-$(CLASSIC_ARCH)
BULK_DIR := $(BUILD_DIR)/bulk-$(BULK_ARCH)

CLASSIC_RAW_LL := $(CLASSIC_DIR)/async_ops.device.raw.ll
CLASSIC_LL := $(CLASSIC_DIR)/async_ops.device.ll
CLASSIC_SC_LL := $(CLASSIC_DIR)/async_ops.device.sc.ll
CLASSIC_PTX := $(CLASSIC_DIR)/async_ops.device.ptx
CLASSIC_CUBIN := $(CLASSIC_DIR)/async_ops.device.cubin
CLASSIC_FATBIN := $(CLASSIC_DIR)/async_ops.fatbin
CLASSIC_HOST_RAW_LL := $(CLASSIC_DIR)/async_ops.host.raw.ll
CLASSIC_HOST_LL := $(CLASSIC_DIR)/async_ops.host.ll
CLASSIC_HOST_OBJ := $(CLASSIC_DIR)/async_ops.host.o
CLASSIC_EXE := $(CLASSIC_DIR)/async_ops

BULK_RAW_LL := $(BULK_DIR)/async_ops.device.raw.ll
BULK_LL := $(BULK_DIR)/async_ops.device.ll
BULK_SC_LL := $(BULK_DIR)/async_ops.device.sc.ll
BULK_PTX := $(BULK_DIR)/async_ops.device.ptx
BULK_CUBIN := $(BULK_DIR)/async_ops.device.cubin

CUDA_BASE_FLAGS := \
	--cuda-path=$(CUDA_HOME) \
	-I$(CUDA_HOME)/include \
	-std=c++17 \
	-gline-tables-only \
	-O0 \
	-Xclang -disable-llvm-passes \
	-Xclang -disable-O0-optnone \
	-fno-discard-value-names

CLASSIC_CUDA_FLAGS := \
	$(CUDA_BASE_FLAGS) \
	--cuda-gpu-arch=$(CLASSIC_ARCH)

BULK_CUDA_FLAGS := \
	$(CUDA_BASE_FLAGS) \
	--cuda-gpu-arch=$(BULK_ARCH)

.PHONY: \
	all \
	classic \
	bulk \
	ir \
	sc \
	ptx \
	cubin \
	run \
	inspect \
	inspect-classic \
	inspect-bulk \
	check \
	check-classic \
	check-bulk \
	info \
	clean

all: classic

classic: $(CLASSIC_EXE)

bulk: $(BULK_CUBIN)

ir: $(CLASSIC_RAW_LL) $(CLASSIC_LL) $(BULK_RAW_LL) $(BULK_LL)

sc: $(CLASSIC_SC_LL)

ptx: $(CLASSIC_PTX)

cubin: $(CLASSIC_CUBIN)

info:
	@echo "SOOPA_COLAIDA_ROOT = $(SOOPA_COLAIDA_ROOT)"
	@echo "LLVM_HOME          = $(LLVM_HOME)"
	@echo "CUDA_HOME          = $(CUDA_HOME)"
	@echo "SC_PLUGIN          = $(SC_PLUGIN)"
	@echo "Detected CC        = $(DETECTED_CC)"
	@echo "CLASSIC_ARCH       = $(CLASSIC_ARCH)"
	@echo "CLASSIC_COMPUTE    = $(CLASSIC_COMPUTE)"
	@echo "BULK_ARCH          = $(BULK_ARCH)"

$(CLASSIC_DIR) $(BULK_DIR):
	mkdir -p $@

$(CLASSIC_RAW_LL): $(SRC) | $(CLASSIC_DIR)
	$(CLANGXX) \
		$(CLASSIC_CUDA_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  CLASSIC RAW LLVM -> $@"

$(CLASSIC_LL): $(CLASSIC_RAW_LL)
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@
	@echo "  CLASSIC LLVM     -> $@"

$(CLASSIC_SC_LL): $(CLASSIC_LL) $(SC_PLUGIN)
	@echo "SuperCollider classic: rdelay=$(SC_RDELAY_NS) wdelay=$(SC_WDELAY_NS) ilu=$(SC_ILU) async=$(SC_ASYNC) bulk=$(SC_BULK) shuffle=$(SC_SHUFFLE) seed=$(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(SC_PASS)' \
		-S \
		$< \
		-o $@
	@echo "  CLASSIC SC LLVM  -> $@"

$(CLASSIC_PTX): $(CLASSIC_SC_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CLASSIC_ARCH) \
		$< \
		-o $@
	@echo "  CLASSIC PTX      -> $@"

$(CLASSIC_CUBIN): $(CLASSIC_PTX)
	$(PTXAS) \
		--gpu-name=$(CLASSIC_ARCH) \
		$< \
		-o $@
	@echo "  CLASSIC CUBIN    -> $@"

$(CLASSIC_FATBIN): $(CLASSIC_CUBIN) $(CLASSIC_PTX)
	$(FATBINARY) \
		--create=$@ \
		--64 \
		--image=profile=$(CLASSIC_ARCH),file=$(CLASSIC_CUBIN) \
		--image=profile=$(CLASSIC_COMPUTE),file=$(CLASSIC_PTX)
	@echo "  CLASSIC FATBIN   -> $@"

$(CLASSIC_HOST_RAW_LL): $(SRC) $(CLASSIC_FATBIN) | $(CLASSIC_DIR)
	$(CLANGXX) \
		$(CLASSIC_CUDA_FLAGS) \
		--cuda-host-only \
		-Xclang -fcuda-include-gpubinary \
		-Xclang $(abspath $(CLASSIC_FATBIN)) \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  CLASSIC HOST RAW -> $@"

$(CLASSIC_HOST_LL): $(CLASSIC_HOST_RAW_LL)
	$(OPT) \
		-passes=mem2reg \
		-S \
		$< \
		-o $@
	@echo "  CLASSIC HOST LLVM -> $@"

$(CLASSIC_HOST_OBJ): $(CLASSIC_HOST_LL)
	$(CLANGXX) \
		-c \
		$< \
		-o $@
	@echo "  CLASSIC HOST OBJ -> $@"

$(CLASSIC_EXE): $(CLASSIC_HOST_OBJ)
	$(CLANGXX) \
		$< \
		-L$(CUDA_HOME)/lib64 \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-lcudart \
		-o $@
	@echo "  CLASSIC EXE      -> $@"

$(BULK_RAW_LL): $(SRC) | $(BULK_DIR)
	$(CLANGXX) \
		$(BULK_CUDA_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@
	@echo "  BULK RAW LLVM -> $@"

$(BULK_LL): $(BULK_RAW_LL)
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@
	@echo "  BULK LLVM     -> $@"

$(BULK_SC_LL): $(BULK_LL) $(SC_PLUGIN)
	@echo "SuperCollider bulk: rdelay=$(SC_RDELAY_NS) wdelay=$(SC_WDELAY_NS) ilu=$(SC_ILU) async=$(SC_ASYNC) bulk=1 shuffle=$(SC_SHUFFLE) seed=$(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(BULK_SC_PASS)' \
		-S \
		$< \
		-o $@
	@echo "  BULK SC LLVM  -> $@"

$(BULK_PTX): $(BULK_SC_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(BULK_ARCH) \
		-mattr=+ptx80 \
		$< \
		-o $@
	@echo "  BULK PTX      -> $@"

$(BULK_CUBIN): $(BULK_PTX)
	$(PTXAS) \
		--gpu-name=$(BULK_ARCH) \
		$< \
		-o $@
	@echo "  BULK CUBIN    -> $@"

run: $(CLASSIC_EXE)
	./$(CLASSIC_EXE)

inspect: inspect-classic

inspect-classic: $(CLASSIC_SC_LL) $(CLASSIC_PTX)
	@echo "=== $(CLASSIC_ARCH) O1 LLVM IR: classic cp.async ==="
	@grep -nE 'cp\.async\.(ca|cg)\.shared\.global|cp\.async\.commit_group|cp\.async\.wait_group|sc\.async|sc_err' $(CLASSIC_SC_LL) || true
	@echo
	@echo "=== $(CLASSIC_ARCH) PTX: classic cp.async ==="
	@grep -nE 'cp\.async\.(ca|cg)\.shared\.global|cp\.async\.commit_group|cp\.async\.wait_group' $(CLASSIC_PTX) || true

inspect-bulk: $(BULK_SC_LL) $(BULK_PTX)
	@echo "=== $(BULK_ARCH) O1 LLVM IR: cp.async.bulk shared -> global ==="
	@grep -nE 'cp\.async\.bulk|fence\.proxy\.async|sc\.async|sc_err' $(BULK_SC_LL) || true
	@echo
	@echo "=== $(BULK_ARCH) PTX: cp.async.bulk shared -> global ==="
	@grep -nE 'cp\.async\.bulk|fence\.proxy\.async' $(BULK_PTX) || true

check: check-classic

check-classic: $(CLASSIC_PTX)
	@grep -Eq 'cp\.async[^;]*,[[:space:]]*4;' $(CLASSIC_PTX) || \
		{ echo "ERROR: 4-byte classic cp.async not found"; exit 1; }
	@grep -Eq 'cp\.async[^;]*,[[:space:]]*8;' $(CLASSIC_PTX) || \
		{ echo "ERROR: 8-byte classic cp.async not found"; exit 1; }
	@grep -Eq 'cp\.async[^;]*,[[:space:]]*16;' $(CLASSIC_PTX) || \
		{ echo "ERROR: 16-byte classic cp.async not found"; exit 1; }
	@grep -q 'cp\.async\.commit_group' $(CLASSIC_PTX) || \
		{ echo "ERROR: classic cp.async.commit_group not found"; exit 1; }
	@grep -q 'cp\.async\.wait_group' $(CLASSIC_PTX) || \
		{ echo "ERROR: classic cp.async.wait_group not found"; exit 1; }
	@echo "PASS: classic cp.async 4/8/16 forms found for $(CLASSIC_ARCH)."

check-bulk: $(BULK_PTX)
	@grep -q 'cp\.async\.bulk\.global\.shared::cta\.bulk_group' $(BULK_PTX) || \
		{ echo "ERROR: shared->global cp.async.bulk form not found"; exit 1; }
	@grep -q 'cp\.async\.bulk\.commit_group' $(BULK_PTX) || \
		{ echo "ERROR: cp.async.bulk.commit_group not found"; exit 1; }
	@grep -q 'cp\.async\.bulk\.wait_group' $(BULK_PTX) || \
		{ echo "ERROR: cp.async.bulk.wait_group not found"; exit 1; }
	@echo "PASS: cp.async.bulk shared->global form found for $(BULK_ARCH)."

clean:
	rm -rf $(BUILD_DIR)