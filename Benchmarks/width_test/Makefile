COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT ?= $(abspath $(COMMON_DIR)/../..)

LLVM_HOME ?= $(ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= $(ROOT)/dependencies/cuda-12.1
SC_PLUGIN ?= $(ROOT)/project/build/SuperColliderPass.so

DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null | head -1 | tr -d '.[:space:]')

ifeq ($(DETECTED_CC),)
$(error Could not detect GPU compute capability)
endif

CUDA_ARCH ?= sm_$(DETECTED_CC)

SC_RDELAY_NS ?= 200
SC_WDELAY_NS ?= 200
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

SC_SHUFFLE_SEED_ARG := $(if $(SC_SHUFFLE_SEED),;shuffle_seed=$(SC_SHUFFLE_SEED),)
SC_PASS := supercollider<rdelay=$(SC_RDELAY_NS);wdelay=$(SC_WDELAY_NS);ilu=$(SC_ILU);async=$(SC_ASYNC);bulk=$(SC_BULK);shuffle=$(SC_SHUFFLE)$(SC_SHUFFLE_SEED_ARG)>

CLANGXX := $(LLVM_HOME)/bin/clang++
OPT := $(LLVM_HOME)/bin/opt
LLC := $(LLVM_HOME)/bin/llc
PTXAS := $(CUDA_HOME)/bin/ptxas

TYPES := u8 u16 u32 u64 f32 f64
SPACES := global shared generic
TESTS := $(foreach space,$(SPACES),$(foreach type,$(TYPES),$(space)_$(type)))
BUILD_DIR := build

RAW_LL := $(TESTS:%=$(BUILD_DIR)/%.raw.ll)
OPT_LL := $(TESTS:%=$(BUILD_DIR)/%.ll)
SC_LL := $(TESTS:%=$(BUILD_DIR)/%.sc.ll)
PTX := $(TESTS:%=$(BUILD_DIR)/%.ptx)
CUBIN := $(TESTS:%=$(BUILD_DIR)/%.cubin)

.PHONY: all raw opt sc verify ptx cubin global shared generic inspect inspect-global inspect-shared inspect-generic clean

all: cubin

raw: $(RAW_LL)
opt: $(OPT_LL)
sc: $(SC_LL)
ptx: $(PTX)
cubin: $(CUBIN)
global: $(filter $(BUILD_DIR)/global_%.cubin,$(CUBIN))
shared: $(filter $(BUILD_DIR)/shared_%.cubin,$(CUBIN))
generic: $(filter $(BUILD_DIR)/generic_%.cubin,$(CUBIN))

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/global_%.raw.ll: global/%/test.cu | $(BUILD_DIR)
	$(CLANGXX) \
		-x cuda \
		--cuda-device-only \
		--cuda-gpu-arch=$(CUDA_ARCH) \
		--cuda-path=$(CUDA_HOME) \
		-O0 \
		-Xclang -disable-llvm-passes \
		-Xclang -disable-O0-optnone \
		-S -emit-llvm \
		$< \
		-o $@

$(BUILD_DIR)/shared_%.raw.ll: shared/%/test.cu | $(BUILD_DIR)
	$(CLANGXX) \
		-x cuda \
		--cuda-device-only \
		--cuda-gpu-arch=$(CUDA_ARCH) \
		--cuda-path=$(CUDA_HOME) \
		-O0 \
		-Xclang -disable-llvm-passes \
		-Xclang -disable-O0-optnone \
		-S -emit-llvm \
		$< \
		-o $@

$(BUILD_DIR)/generic_%.raw.ll: generic/%/test.cu | $(BUILD_DIR)
	$(CLANGXX) \
		-x cuda \
		--cuda-device-only \
		--cuda-gpu-arch=$(CUDA_ARCH) \
		--cuda-path=$(CUDA_HOME) \
		-O0 \
		-Xclang -disable-llvm-passes \
		-Xclang -disable-O0-optnone \
		-S -emit-llvm \
		$< \
		-o $@

$(BUILD_DIR)/%.ll: $(BUILD_DIR)/%.raw.ll
	$(OPT) \
		-passes='default<O1>' \
		-S \
		$< \
		-o $@

$(BUILD_DIR)/%.sc.ll: $(BUILD_DIR)/%.ll
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(SC_PASS)' \
		-S \
		$< \
		-o $@ \
		2> $(BUILD_DIR)/$*.sc.log

verify: $(SC_LL)
	@set -e; \
	for test in $(TESTS); do \
		echo "===== verify $$test ====="; \
		$(OPT) -passes=verify -disable-output $(BUILD_DIR)/$$test.sc.ll; \
	done

$(BUILD_DIR)/%.ptx: $(BUILD_DIR)/%.sc.ll
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CUDA_ARCH) \
		$< \
		-o $@

$(BUILD_DIR)/%.cubin: $(BUILD_DIR)/%.ptx
	$(PTXAS) \
		-arch=$(CUDA_ARCH) \
		$< \
		-o $@

inspect: inspect-global inspect-shared inspect-generic

inspect-global: $(filter $(BUILD_DIR)/global_%,$(SC_LL)) $(filter $(BUILD_DIR)/global_%,$(PTX))
	@for type in $(TYPES); do \
		test=global_$$type; \
		echo; \
		echo "===== $$test : LLVM ====="; \
		grep -n -C 1 'sc.test.strong' $(BUILD_DIR)/$$test.sc.ll || true; \
		echo "===== $$test : PTX ====="; \
		grep -n -E 'ld\.relaxed\.sys|ld\.global|st\.global' $(BUILD_DIR)/$$test.ptx || true; \
	done

inspect-shared: $(filter $(BUILD_DIR)/shared_%,$(SC_LL)) $(filter $(BUILD_DIR)/shared_%,$(PTX))
	@for type in $(TYPES); do \
		test=shared_$$type; \
		echo; \
		echo "===== $$test : LLVM ====="; \
		grep -n -C 1 'sc.test.strong' $(BUILD_DIR)/$$test.sc.ll || true; \
		echo "===== $$test : PTX ====="; \
		grep -n -E 'ld\.relaxed\.sys|ld\.shared|st\.shared' $(BUILD_DIR)/$$test.ptx || true; \
	done

inspect-generic: $(filter $(BUILD_DIR)/generic_%,$(SC_LL)) $(filter $(BUILD_DIR)/generic_%,$(PTX))
	@for type in $(TYPES); do \
		test=generic_$$type; \
		echo; \
		echo "===== $$test : LLVM ====="; \
		grep -n -C 1 'sc.test.strong' $(BUILD_DIR)/$$test.sc.ll || true; \
		echo "===== $$test : PTX ====="; \
		grep -n -E 'ld\.relaxed\.sys|ld\.global|st\.global' $(BUILD_DIR)/$$test.ptx || true; \
	done

clean:
	rm -rf $(BUILD_DIR)