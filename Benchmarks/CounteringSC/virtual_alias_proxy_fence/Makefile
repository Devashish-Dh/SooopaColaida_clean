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
PTX_VERSION ?= 7.5

SC_RDELAY_NS ?= 5000
SC_WDELAY_NS ?= 1
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

NAME := virtual_alias_proxy_fence
SRC := $(NAME).cu
BUILD := build-artifacts

DEVICE_RAW_LL := $(BUILD)/$(NAME).device.raw.ll
DEVICE_LL := $(BUILD)/$(NAME).device.ll
DEVICE_SC_LL := $(BUILD)/$(NAME).device.sc.ll
DEVICE_PTX := $(BUILD)/$(NAME).device.ptx
DEVICE_CUBIN := $(BUILD)/$(NAME).device.cubin
FATBIN := $(BUILD)/$(NAME).fatbin
HOST_RAW_LL := $(BUILD)/$(NAME).host.raw.ll
HOST_LL := $(BUILD)/$(NAME).host.ll
HOST_OBJ := $(BUILD)/$(NAME).host.o
EXE := $(BUILD)/$(NAME)
RUN_LOG := $(BUILD)/$(NAME).run.log

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

.PHONY: all clean run check-ir check-ptx print-config

all: $(EXE)

print-config:
	@echo "SuperCollider virtual-alias proxy-fence counterexample:"
	@echo "  plugin  = $(SC_PLUGIN)"
	@echo "  runtime = $(SC_RUNTIME)"
	@echo "  rdelay  = $(SC_RDELAY_NS) ns"
	@echo "  wdelay  = $(SC_WDELAY_NS) ns"
	@echo "  ilu     = $(SC_ILU)"
	@echo "  async   = $(SC_ASYNC)"
	@echo "  bulk    = $(SC_BULK)"
	@echo "  shuffle = $(SC_SHUFFLE)"
	@echo "  seed    = $(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"

$(BUILD):
	@mkdir -p $(BUILD)

$(DEVICE_RAW_LL): $(SRC) | $(BUILD)
	$(CLANGXX) $(CUDA_COMMON) \
		--cuda-device-only \
		-S -emit-llvm \
		$(SRC) \
		-o $@

$(DEVICE_LL): $(DEVICE_RAW_LL)
	$(OPT) \
		-passes='default<O1>' \
		-S $< \
		-o $@

$(DEVICE_SC_LL): $(DEVICE_LL)
	@test -f "$(SC_PLUGIN)" || { echo "Missing SuperCollider pass: $(SC_PLUGIN)"; exit 1; }
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(SC_PASS)' \
		-S $< \
		-o $@ \
		2> $(BUILD)/$(NAME).sc.log

$(DEVICE_PTX): $(DEVICE_SC_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CUDA_ARCH) \
		$< \
		-o $@
	@sed -i 's/^\.version .*/.version $(PTX_VERSION)/' $@

$(DEVICE_CUBIN): $(DEVICE_PTX)
	$(PTXAS) \
		--gpu-name=$(CUDA_ARCH) \
		$< \
		-o $@

$(FATBIN): $(DEVICE_CUBIN) $(DEVICE_PTX)
	$(FATBINARY) \
		--create=$@ \
		--64 \
		--image=profile=$(CUDA_ARCH),file=$(DEVICE_CUBIN) \
		--image=profile=$(CUDA_COMPUTE_ARCH),file=$(DEVICE_PTX)

$(HOST_RAW_LL): $(SRC) $(FATBIN)
	$(CLANGXX) $(CUDA_COMMON) \
		--cuda-host-only \
		-Xclang -fcuda-include-gpubinary \
		-Xclang $(abspath $(FATBIN)) \
		-S -emit-llvm \
		$(SRC) \
		-o $@

$(HOST_LL): $(HOST_RAW_LL)
	$(OPT) \
		-passes=mem2reg \
		-S $< \
		-o $@

$(HOST_OBJ): $(HOST_LL)
	$(CLANGXX) -c $< -o $@

$(EXE): $(HOST_OBJ)
	$(CLANGXX) \
		$< \
		-L$(CUDA_HOME)/lib64 \
		-L$(CUDA_HOME)/lib64/stubs \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-lcudart \
		-lcuda \
		-o $@

check-ir: $(DEVICE_SC_LL)
	@echo "=== virtual-alias proxy-fence LLVM IR ==="
	@grep -nE 'alias_no_fence|alias_single_thread_with_fence|fence\.proxy\.alias|sc_err|sc_err_ilu|sc\.ilu\.|sc\.test\.strong|llvm\.nvvm\.read\.ptx\.sreg\.laneid' $(DEVICE_SC_LL) || true

check-ptx: $(DEVICE_PTX)
	@echo "=== virtual-alias proxy-fence PTX ==="
	@grep -nE 'alias_no_fence|alias_single_thread_with_fence|fence\.proxy\.alias|ld\.global|st\.global|nanosleep|laneid|activemask' $(DEVICE_PTX) || true

run: $(EXE)
	@test -f "$(SC_RUNTIME)" || { echo "Missing SuperCollider runtime: $(SC_RUNTIME)"; exit 1; }
	@set -o pipefail; \
	LD_PRELOAD=$(SC_RUNTIME) \
	LD_LIBRARY_PATH=$(CUDA_HOME)/extras/CUPTI/lib64:$(CUDA_HOME)/targets/x86_64-linux/lib:$(SC_PROJECT)/build:$(CUDA_HOME)/lib64:$(CUDA_HOME)/lib64/stubs:$(LLVM_HOME)/lib:$${LD_LIBRARY_PATH} \
	$(EXE) 2>&1 | tee $(RUN_LOG)

clean:
	rm -rf $(BUILD)
