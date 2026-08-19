SHELL := /bin/bash

COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOOPA_COLAIDA_ROOT ?= $(abspath $(COMMON_DIR)/../../..)
LLVM_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= /disk/devashish/CUDA_Toolkits/cuda-12.9
SC_PROJECT ?= $(SOOPA_COLAIDA_ROOT)/project

CLANGXX ?= $(LLVM_HOME)/bin/clang++
OPT ?= $(LLVM_HOME)/bin/opt
LLC ?= $(LLVM_HOME)/bin/llc
PTXAS ?= $(CUDA_HOME)/bin/ptxas
FATBINARY ?= $(CUDA_HOME)/bin/fatbinary

SC_PLUGIN ?= $(SC_PROJECT)/build/SuperColliderPass.so
SC_RUNTIME ?= $(SC_PROJECT)/build/libsc.so

CUDA_ARCH ?= sm_86
CUDA_COMPUTE_ARCH := $(subst sm_,compute_,$(CUDA_ARCH))

SC_RDELAY_NS ?= 5000
SC_WDELAY_NS ?= 1
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

ENABLE_CLUSTER ?= 0

BASE_PROGRAMS := \
	cta_cta_same_block_control \
	cta_cta_different_blocks \
	cta_gpu_different_blocks \
	cta_atomic_vs_weak_load \
	weak_store_vs_cta_atomic \
	cta_cas_different_blocks \
	broken_cta_spinlock

CLUSTER_PROGRAMS := cluster_cluster_different_clusters
PROGRAMS := $(BASE_PROGRAMS) $(if $(filter 1,$(ENABLE_CLUSTER)),$(CLUSTER_PROGRAMS),)

BUILD := build-artifacts
EXES := $(PROGRAMS:%=$(BUILD)/%)

CUDA_COMMON := \
	--cuda-path=$(CUDA_HOME) \
	--cuda-gpu-arch=$(CUDA_ARCH) \
	-I$(CUDA_HOME)/include \
	-gline-tables-only \
	-O0 \
	-Xclang -disable-llvm-passes \
	-Xclang -disable-O0-optnone \
	-fno-discard-value-names

SC_SHUFFLE_SEED_ARG := $(if $(SC_SHUFFLE_SEED),;shuffle_seed=$(SC_SHUFFLE_SEED),)
SC_PASS := supercollider<rdelay=$(SC_RDELAY_NS);wdelay=$(SC_WDELAY_NS);ilu=$(SC_ILU);async=$(SC_ASYNC);bulk=$(SC_BULK);shuffle=$(SC_SHUFFLE)$(SC_SHUFFLE_SEED_ARG)>

.PHONY: all clean run check-ir check-ptx print-config cluster

all: $(EXES)

cluster:
	@$(MAKE) --no-print-directory ENABLE_CLUSTER=1 CUDA_ARCH=sm_90 all

print-config:
	@echo "SuperCollider scoped-atomic race suite:"
	@echo "  plugin         = $(SC_PLUGIN)"
	@echo "  runtime        = $(SC_RUNTIME)"
	@echo "  arch           = $(CUDA_ARCH)"
	@echo "  rdelay         = $(SC_RDELAY_NS) ns"
	@echo "  wdelay         = $(SC_WDELAY_NS) ns"
	@echo "  ilu            = $(SC_ILU)"
	@echo "  async          = $(SC_ASYNC)"
	@echo "  bulk           = $(SC_BULK)"
	@echo "  shuffle        = $(SC_SHUFFLE)"
	@echo "  seed           = $(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"
	@echo "  cluster tests  = $(ENABLE_CLUSTER)"
	@echo "  programs       = $(PROGRAMS)"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.device.raw.ll: %.cu | $(BUILD)
	$(CLANGXX) $(CUDA_COMMON) \
		--cuda-device-only \
		-S -emit-llvm \
		$< \
		-o $@

$(BUILD)/%.device.ll: $(BUILD)/%.device.raw.ll
	$(OPT) \
		-passes='default<O1>' \
		-S $< \
		-o $@

$(BUILD)/%.device.sc.ll: $(BUILD)/%.device.ll
	@test -f "$(SC_PLUGIN)" || { echo "Missing SuperCollider pass: $(SC_PLUGIN)"; exit 1; }
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(SC_PASS)' \
		-S $< \
		-o $@ \
		2> $(BUILD)/$*.sc.log

$(BUILD)/%.device.ptx: $(BUILD)/%.device.sc.ll
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CUDA_ARCH) \
		$< \
		-o $@

$(BUILD)/%.device.cubin: $(BUILD)/%.device.ptx
	$(PTXAS) \
		--gpu-name=$(CUDA_ARCH) \
		$< \
		-o $@

$(BUILD)/%.fatbin: $(BUILD)/%.device.cubin $(BUILD)/%.device.ptx
	$(FATBINARY) \
		--create=$@ \
		--64 \
		--image=profile=$(CUDA_ARCH),file=$(BUILD)/$*.device.cubin \
		--image=profile=$(CUDA_COMPUTE_ARCH),file=$(BUILD)/$*.device.ptx

$(BUILD)/%.host.raw.ll: %.cu $(BUILD)/%.fatbin
	$(CLANGXX) $(CUDA_COMMON) \
		--cuda-host-only \
		-Xclang -fcuda-include-gpubinary \
		-Xclang $(abspath $(BUILD)/$*.fatbin) \
		-S -emit-llvm \
		$< \
		-o $@

$(BUILD)/%.host.ll: $(BUILD)/%.host.raw.ll
	$(OPT) \
		-passes=mem2reg \
		-S $< \
		-o $@

$(BUILD)/%.host.o: $(BUILD)/%.host.ll
	$(CLANGXX) -c $< -o $@

$(BUILD)/%: $(BUILD)/%.host.o
	$(CLANGXX) \
		$< \
		-L$(CUDA_HOME)/lib64 \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-lcudart \
		-o $@

check-ir: $(PROGRAMS:%=$(BUILD)/%.device.sc.ll)
	@set -e; \
	for name in $(PROGRAMS); do \
		echo ""; \
		echo "=== $$name: pre-SC atomic/weak ops ==="; \
		grep -nE 'atomicrmw|cmpxchg|load |store |__.*Atomic|__nv_atomic' "$(BUILD)/$$name.device.ll" || true; \
		echo "=== $$name: post-SC instrumentation ==="; \
		grep -nE 'atomicrmw|cmpxchg|sc_err|sc_err_ilu|sc\\.|__.*Atomic|__nv_atomic' "$(BUILD)/$$name.device.sc.ll" || true; \
	done

check-ptx: $(PROGRAMS:%=$(BUILD)/%.device.ptx)
	@set -e; \
	for name in $(PROGRAMS); do \
		echo ""; \
		echo "=== $$name PTX ==="; \
		grep -nE 'atom\\.|red\\.|ld\\.(global|relaxed|acquire|volatile)|st\\.(global|relaxed|release|volatile)|nanosleep|match\\.any' "$(BUILD)/$$name.device.ptx" || true; \
	done

run: $(EXES)
	@test -f "$(SC_RUNTIME)" || { echo "Missing SuperCollider runtime: $(SC_RUNTIME)"; exit 1; }
	@set -e -o pipefail; \
	for name in $(PROGRAMS); do \
		echo ""; \
		echo "=== $$name ==="; \
		LD_PRELOAD=$(SC_RUNTIME) \
		LD_LIBRARY_PATH=$(CUDA_HOME)/extras/CUPTI/lib64:$(CUDA_HOME)/targets/x86_64-linux/lib:$(SC_PROJECT)/build:$(CUDA_HOME)/lib64:$(LLVM_HOME)/lib:$${LD_LIBRARY_PATH} \
		"$(BUILD)/$$name" 2>&1 | tee "$(BUILD)/$$name.run.log"; \
	done

clean:
	rm -rf $(BUILD) run-logs

.PRECIOUS: \
	$(BUILD)/%.device.raw.ll \
	$(BUILD)/%.device.ll \
	$(BUILD)/%.device.sc.ll \
	$(BUILD)/%.device.ptx \
	$(BUILD)/%.device.cubin \
	$(BUILD)/%.fatbin \
	$(BUILD)/%.host.raw.ll \
	$(BUILD)/%.host.ll \
	$(BUILD)/%.host.o
