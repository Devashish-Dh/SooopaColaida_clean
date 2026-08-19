COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT ?= $(abspath $(COMMON_DIR)/../..)

LLVM_HOME ?= $(PROJECT_ROOT)/dependencies/INSTALL-llvm
CUDA_HOME ?= $(PROJECT_ROOT)/dependencies/cuda-12.1

DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null | head -1 | tr -d '.[:space:]')

ifeq ($(DETECTED_CC),)
$(error Could not detect GPU compute capability)
endif

CUDA_ARCH ?= sm_$(DETECTED_CC)
CUDA_COMPUTE_ARCH := $(subst sm_,compute_,$(CUDA_ARCH))

CLANGXX ?= $(LLVM_HOME)/bin/clang++
OPT ?= $(LLVM_HOME)/bin/opt
LLC ?= $(LLVM_HOME)/bin/llc

PTXAS ?= $(CUDA_HOME)/bin/ptxas
FATBINARY ?= $(CUDA_HOME)/bin/fatbinary

OUT ?= build-artifacts
SRC ?= llTest.cu

SRC_NAME := $(basename $(notdir $(SRC)))
SRC_EXT := $(suffix $(SRC))

BIN := $(OUT)/$(SRC_NAME)

HOST_RAW_LL := $(OUT)/$(SRC_NAME).host.raw.ll
HOST_LL := $(OUT)/$(SRC_NAME).host.ll
HOST_OBJ := $(OUT)/$(SRC_NAME).host.o

DEVICE_RAW_LL := $(OUT)/$(SRC_NAME).device.raw.ll
DEVICE_LL := $(OUT)/$(SRC_NAME).device.ll
DEVICE_SC_LL := $(OUT)/$(SRC_NAME).device.sc.ll
DEVICE_PTX := $(OUT)/$(SRC_NAME).device.ptx
DEVICE_CUBIN := $(OUT)/$(SRC_NAME).device.cubin
DEVICE_FATBIN := $(OUT)/$(SRC_NAME).fatbin

SUPERCOLLIDER_SO ?= $(PROJECT_ROOT)/project/build/SuperColliderPass.so

SC_RDELAY_NS ?= 200
SC_WDELAY_NS ?= 200
SC_ILU ?= 1
SC_ASYNC ?= 1
SC_BULK ?= 0
SC_SHUFFLE ?= 1
SC_SHUFFLE_SEED ?= 12345

SC_SHUFFLE_SEED_ARG := $(if $(SC_SHUFFLE_SEED),;shuffle_seed=$(SC_SHUFFLE_SEED),)
SC_PASS := supercollider<rdelay=$(SC_RDELAY_NS);wdelay=$(SC_WDELAY_NS);ilu=$(SC_ILU);async=$(SC_ASYNC);bulk=$(SC_BULK);shuffle=$(SC_SHUFFLE)$(SC_SHUFFLE_SEED_ARG)>

CUDA_FLAGS := \
	--cuda-path=$(CUDA_HOME) \
	--cuda-gpu-arch=$(CUDA_ARCH)

CUDA_CPPFLAGS := \
	-I$(CUDA_HOME)/include

LLVM_IR_FLAGS := \
	-gline-tables-only \
	-O0 \
	-Xclang -disable-llvm-passes \
	-Xclang -disable-O0-optnone \
	-fno-discard-value-names

DEVICE_OPT_PIPELINE ?= default<O1>

MEM2REG_FLAGS := -passes=mem2reg

CUDA_LDFLAGS := \
	-L$(CUDA_HOME)/lib64 \
	-Wl,-rpath,$(CUDA_HOME)/lib64

CUDA_LDLIBS := -lcudart

.PHONY: all clean

all: $(BIN)

$(OUT):
	mkdir -p $@

ifeq ($(SRC_EXT),.cu)

$(DEVICE_RAW_LL): $(SRC) | $(OUT)
	$(CLANGXX) \
		$(CUDA_FLAGS) \
		$(CUDA_CPPFLAGS) \
		$(LLVM_IR_FLAGS) \
		--cuda-device-only \
		-S \
		-emit-llvm \
		$< \
		-o $@

$(DEVICE_LL): $(DEVICE_RAW_LL)
	$(OPT) \
		-passes='$(DEVICE_OPT_PIPELINE)' \
		-S \
		$< \
		-o $@

$(DEVICE_SC_LL): $(DEVICE_LL)
	@test -f "$(SUPERCOLLIDER_SO)" || \
		(echo "SuperCollider pass not found: $(SUPERCOLLIDER_SO)" && exit 1)
	@echo ""
	@echo "SuperCollider:"
	@echo "  rdelay = $(SC_RDELAY_NS) ns"
	@echo "  wdelay = $(SC_WDELAY_NS) ns"
	@echo "  ilu    = $(SC_ILU)"
	@echo "  async  = $(SC_ASYNC)"
	@echo "  bulk   = $(SC_BULK)"
	@echo "  shuffle = $(SC_SHUFFLE)"
	@echo "  seed   = $(if $(SC_SHUFFLE_SEED),$(SC_SHUFFLE_SEED),<default>)"
	$(OPT) \
		-load-pass-plugin=$(SUPERCOLLIDER_SO) \
		-passes='$(SC_PASS)' \
		-S \
		$< \
		-o $@
	@echo "  LLVM IR -> $(DEVICE_SC_LL)"

$(DEVICE_PTX): $(DEVICE_SC_LL)
	$(LLC) \
		-march=nvptx64 \
		-mcpu=$(CUDA_ARCH) \
		$< \
		-o $@
	@echo "  PTX     -> $(DEVICE_PTX)"

$(DEVICE_CUBIN): $(DEVICE_PTX)
	$(PTXAS) \
		--gpu-name=$(CUDA_ARCH) \
		$< \
		-o $@
	@echo "  CUBIN   -> $(DEVICE_CUBIN)"

$(DEVICE_FATBIN): $(DEVICE_CUBIN) $(DEVICE_PTX)
	$(FATBINARY) \
		--create=$@ \
		--64 \
		--image=profile=$(CUDA_ARCH),file=$(DEVICE_CUBIN) \
		--image=profile=$(CUDA_COMPUTE_ARCH),file=$(DEVICE_PTX)

$(HOST_RAW_LL): $(SRC) $(DEVICE_FATBIN) | $(OUT)
	$(CLANGXX) \
		$(CUDA_FLAGS) \
		$(CUDA_CPPFLAGS) \
		$(LLVM_IR_FLAGS) \
		--cuda-host-only \
		-Xclang -fcuda-include-gpubinary \
		-Xclang $(abspath $(DEVICE_FATBIN)) \
		-S \
		-emit-llvm \
		$< \
		-o $@

$(HOST_LL): $(HOST_RAW_LL)
	$(OPT) \
		$(MEM2REG_FLAGS) \
		-S \
		$< \
		-o $@

$(HOST_OBJ): $(HOST_LL)
	$(CLANGXX) \
		-c \
		$< \
		-o $@

$(BIN): $(HOST_OBJ)
	$(CLANGXX) \
		$(HOST_OBJ) \
		$(CUDA_LDFLAGS) \
		$(CUDA_LDLIBS) \
		-o $@
	@echo "  EXE     -> $(BIN)"
	@echo ""

else

$(HOST_RAW_LL): $(SRC) | $(OUT)
	$(CLANGXX) \
		$(LLVM_IR_FLAGS) \
		-S \
		-emit-llvm \
		$< \
		-o $@

$(HOST_LL): $(HOST_RAW_LL)
	$(OPT) \
		$(MEM2REG_FLAGS) \
		-S \
		$< \
		-o $@

$(HOST_OBJ): $(HOST_LL)
	$(CLANGXX) \
		-c \
		$< \
		-o $@

$(BIN): $(HOST_OBJ)
	$(CLANGXX) \
		$(HOST_OBJ) \
		-o $@

endif

clean:
	rm -rf $(OUT)