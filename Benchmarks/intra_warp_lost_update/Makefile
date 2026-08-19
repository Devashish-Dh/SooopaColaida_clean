SHELL := /bin/bash

COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOOPA_COLAIDA_ROOT ?= $(abspath $(COMMON_DIR)/../..)

LLVM_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= $(SOOPA_COLAIDA_ROOT)/dependencies/cuda-12.1
SC_PROJECT ?= $(SOOPA_COLAIDA_ROOT)/project

DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null | head -1 | tr -d '.[:space:]')

ifeq ($(DETECTED_CC),)
$(error Could not detect GPU compute capability)
endif

CUDA_ARCH ?= sm_$(DETECTED_CC)
CUDA_COMPUTE ?= compute_$(DETECTED_CC)

CLANGXX := $(LLVM_HOME)/bin/clang++
OPT := $(LLVM_HOME)/bin/opt
LLC := $(LLVM_HOME)/bin/llc
PTXAS := $(CUDA_HOME)/bin/ptxas
FATBINARY := $(CUDA_HOME)/bin/fatbinary

SC_PLUGIN ?= $(SC_PROJECT)/build/SuperColliderPass.so
SC_RUNTIME ?= $(SC_PROJECT)/build/libsc.so

SC_RDELAY_NS ?= 1
SC_WDELAY_NS ?= 1
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

NAME := intra_warp_lost_update
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

.PHONY: all clean run validate check-ir check-ptx print-config

all: $(EXE)

print-config:
	@echo "SuperCollider ILU test:"
	@echo "  arch    = $(CUDA_ARCH)"
	@echo "  compute = $(CUDA_COMPUTE)"
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
	mkdir -p $(BUILD)

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

$(DEVICE_SC_LL): $(DEVICE_LL) $(SC_PLUGIN)
	@echo
	@echo "SuperCollider: rdelay=$(SC_RDELAY_NS) wdelay=$(SC_WDELAY_NS) ilu=$(SC_ILU) async=$(SC_ASYNC) bulk=$(SC_BULK) shuffle=$(SC_SHUFFLE) seed=$(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"
	$(OPT) \
		-load-pass-plugin=$(SC_PLUGIN) \
		-passes='$(SC_PASS)' \
		-S $< \
		-o $@

$(DEVICE_PTX): $(DEVICE_SC_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CUDA_ARCH) \
		$< \
		-o $@

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
		--image=profile=$(CUDA_COMPUTE),file=$(DEVICE_PTX)

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
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-lcudart \
		-o $@

check-ir: $(DEVICE_SC_LL)
	@echo "=== ILU LLVM IR ==="
	@grep -nE 'llvm\.nvvm\.match\.any\.sync\.i64|llvm\.nvvm\.read\.ptx\.sreg\.laneid|llvm\.nvvm\.isspacep\.local|llvm\.nvvm\.vote\.ballot\.sync|llvm\.ctpop\.i32|llvm\.cttz\.i32|sc_err_ilu|sc\.ilu\.' $(DEVICE_SC_LL) || true
	@echo
	@echo "ILU match calls:"
	@grep -c 'llvm.nvvm.match.any.sync.i64' $(DEVICE_SC_LL) || true
	@echo "ILU leader selections:"
	@grep -c 'llvm.cttz.i32' $(DEVICE_SC_LL) || true

check-ptx: $(DEVICE_PTX)
	@echo "=== ILU PTX ==="
	@grep -nE 'match\.any\.sync|popc|activemask|vote\.ballot\.sync|isspacep|laneid|brev|clz|bfind|fns' $(DEVICE_PTX) || true

run: $(EXE)
	@test -f $(SC_RUNTIME) || { echo "Missing runtime: $(SC_RUNTIME)"; exit 1; }
	LD_PRELOAD=$(SC_RUNTIME) \
	LD_LIBRARY_PATH=$(CUDA_HOME)/extras/CUPTI/lib64:$(CUDA_HOME)/targets/x86_64-linux/lib:$(SC_PROJECT)/build:$(CUDA_HOME)/lib64:$(LLVM_HOME)/lib:$${LD_LIBRARY_PATH} \
	$(EXE)

validate: $(EXE)
	@test "$(SC_ILU)" = "1" || { echo "validate requires SC_ILU=1"; exit 1; }
	@test -f $(SC_RUNTIME) || { echo "Missing runtime: $(SC_RUNTIME)"; exit 1; }
	@set -o pipefail; \
	LD_PRELOAD=$(SC_RUNTIME) \
	LD_LIBRARY_PATH=$(CUDA_HOME)/extras/CUPTI/lib64:$(CUDA_HOME)/targets/x86_64-linux/lib:$(SC_PROJECT)/build:$(CUDA_HOME)/lib64:$(LLVM_HOME)/lib:$${LD_LIBRARY_PATH} \
	$(EXE) 2>&1 | tee $(RUN_LOG)
	@grep -q 'This warp WROTE more than one value to the same address' $(RUN_LOG) || { \
		echo "ERROR: no ILU runtime report was found"; \
		exit 1; \
	}
	@grep -q 'ilu_global_same_value_kernel' $(RUN_LOG) || { \
		echo "ERROR: global positive ILU site was not reported"; \
		exit 1; \
	}
	@grep -q 'ilu_shared_same_value_kernel' $(RUN_LOG) || { \
		echo "ERROR: shared positive ILU site was not reported"; \
		exit 1; \
	}
	@grep -q 'ilu_generic_global_same_value_kernel' $(RUN_LOG) || { \
		echo "ERROR: generic->global positive ILU site was not reported"; \
		exit 1; \
	}
	@if grep -q '! In function ilu_unique_address_negative_kernel' $(RUN_LOG); then \
		echo "ERROR: unique-address negative control produced a race report"; \
		exit 1; \
	fi
	@if grep -q '! In function generic_local_store_noinline' $(RUN_LOG); then \
		echo "ERROR: generic->local negative control produced a race report"; \
		exit 1; \
	fi
	@echo "ILU validation passed."

clean:
	rm -rf $(BUILD)