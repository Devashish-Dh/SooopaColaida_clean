cmake_minimum_required(VERSION 3.20)

project(SuperCollider LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOOPA_COLAIDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

# LLVM
set(SOOPA_COLAIDA_LLVM_DIR
    "${SOOPA_COLAIDA_ROOT}/dependencies/INSTALL-llvm/lib/cmake/llvm")

if(NOT DEFINED LLVM_DIR AND EXISTS "${SOOPA_COLAIDA_LLVM_DIR}/LLVMConfig.cmake")
    set(LLVM_DIR "${SOOPA_COLAIDA_LLVM_DIR}" CACHE PATH "Path to LLVMConfig.cmake")
endif()

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

# CUDA
set(SC_CUDA_HOME "$ENV{CUDA_HOME}"
    CACHE PATH "CUDA toolkit used to compile the SuperCollider device callback")

if(NOT SC_CUDA_HOME)
    message(FATAL_ERROR
        "CUDA_HOME is not set. Source ACTIVATE_ENV.sh before building.")
endif()

if(NOT EXISTS "${SC_CUDA_HOME}/include/cuda.h")
    message(FATAL_ERROR
        "CUDA toolkit not found under SC_CUDA_HOME=${SC_CUDA_HOME}")
endif()

# CUDA architecture.
# SC_CUDA_ARCH can be explicitly provided, otherwise detect GPU 0.
if(DEFINED ENV{SC_CUDA_ARCH} AND NOT "$ENV{SC_CUDA_ARCH}" STREQUAL "")
    set(SC_CUDA_ARCH "$ENV{SC_CUDA_ARCH}")
else()
    execute_process(
        COMMAND nvidia-smi
            --query-gpu=compute_cap
            --format=csv,noheader
            --id=0
        OUTPUT_VARIABLE SC_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE SC_NVIDIA_SMI_RESULT
    )

    if(NOT SC_NVIDIA_SMI_RESULT EQUAL 0 OR SC_COMPUTE_CAP STREQUAL "")
        message(FATAL_ERROR
            "Could not detect CUDA compute capability with nvidia-smi. "
            "Set SC_CUDA_ARCH manually, e.g. SC_CUDA_ARCH=sm_86.")
    endif()

    string(REPLACE "." "" SC_COMPUTE_CAP "${SC_COMPUTE_CAP}")
    set(SC_CUDA_ARCH "sm_${SC_COMPUTE_CAP}")
endif()

message(STATUS "CUDA toolkit: ${SC_CUDA_HOME}")
message(STATUS "CUDA architecture: ${SC_CUDA_ARCH}")

# LLVM tools
set(SC_CLANGXX "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(SC_OPT "${LLVM_TOOLS_BINARY_DIR}/opt")

if(NOT EXISTS "${SC_CLANGXX}")
    message(FATAL_ERROR "LLVM clang++ not found: ${SC_CLANGXX}")
endif()

if(NOT EXISTS "${SC_OPT}")
    message(FATAL_ERROR "LLVM opt not found: ${SC_OPT}")
endif()

# LLVM linker library
get_filename_component(SC_LLVM_LIB_DIR "${LLVM_DIR}/../.." ABSOLUTE)

find_library(SC_LLVM_LINKER_ARCHIVE
    NAMES LLVMLinker
    HINTS "${SC_LLVM_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(NOT SC_LLVM_LINKER_ARCHIVE)
    message(FATAL_ERROR
        "LLVM Linker archive was not found under ${SC_LLVM_LIB_DIR}")
endif()

# Device runtime
set(SC_DEVICE_RUNTIME_RAW_LL
    "${CMAKE_BINARY_DIR}/sc_error_device.raw.ll")

set(SC_DEVICE_RUNTIME_LL
    "${CMAKE_BINARY_DIR}/sc_error_device.ll")

add_custom_command(
    OUTPUT "${SC_DEVICE_RUNTIME_RAW_LL}"

    COMMAND "${SC_CLANGXX}"
        --cuda-path=${SC_CUDA_HOME}
        --cuda-gpu-arch=${SC_CUDA_ARCH}
        -I${SC_CUDA_HOME}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
        -O0
        -Xclang -disable-llvm-passes
        -Xclang -disable-O0-optnone
        -fno-discard-value-names
        --cuda-device-only
        -S
        -emit-llvm
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/sc_error_device.cu"
        -o "${SC_DEVICE_RUNTIME_RAW_LL}"

    DEPENDS
        runtime/sc_error_device.cu
        include/sc_runtime_abi.h

    VERBATIM
)

add_custom_command(
    OUTPUT "${SC_DEVICE_RUNTIME_LL}"

    COMMAND "${SC_OPT}"
        -passes=default<O1>
        -S
        "${SC_DEVICE_RUNTIME_RAW_LL}"
        -o "${SC_DEVICE_RUNTIME_LL}"

    DEPENDS
        "${SC_DEVICE_RUNTIME_RAW_LL}"

    VERBATIM
)

add_custom_target(SuperColliderDeviceRuntime
    DEPENDS "${SC_DEVICE_RUNTIME_LL}")

# SuperCollider LLVM pass
add_llvm_pass_plugin(SuperColliderPass
    src/plugin_pass.cpp
    src/async_memops.cpp
    src/block_shuffle.cpp
    src/config.cpp
    src/debugging_prints.cpp
    src/delay.cpp
    src/filter_memops.cpp
    src/fnv1a_hash.cpp
    src/instrument_memops.cpp
    src/intra_warp_lost_update.cpp
    src/parsing_nvvm_ir.cpp
    src/race_assert.cpp
    src/sc_error.cpp
    src/strong_memops.cpp
)

add_dependencies(SuperColliderPass SuperColliderDeviceRuntime)

target_include_directories(SuperColliderPass
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${LLVM_INCLUDE_DIRS}"
)

target_compile_options(SuperColliderPass
    PRIVATE
        ${LLVM_DEFINITIONS_LIST}
)

target_compile_definitions(SuperColliderPass
    PRIVATE
        SC_DEVICE_RUNTIME_LL_PATH="${SC_DEVICE_RUNTIME_LL}"
)

target_link_libraries(SuperColliderPass
    PRIVATE
        "${SC_LLVM_LINKER_ARCHIVE}"
)

set_target_properties(SuperColliderPass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "SuperColliderPass"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Host-side reporting runtime.
set(SC_CUPTI_INCLUDE_DIR
    "${SC_CUDA_HOME}/extras/CUPTI/include")

find_library(SC_CUPTI_LIBRARY
    NAMES cupti
    HINTS
        "${SC_CUDA_HOME}/extras/CUPTI/lib64"
        "${SC_CUDA_HOME}/lib64"
)

# libcuda belongs to the NVIDIA driver, not the CUDA toolkit.
find_library(SC_CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        "${SC_CUDA_HOME}/lib64/stubs"
)

if(NOT EXISTS "${SC_CUPTI_INCLUDE_DIR}/cupti.h")
    message(FATAL_ERROR
        "CUPTI headers not found under ${SC_CUPTI_INCLUDE_DIR}")
endif()

if(NOT SC_CUPTI_LIBRARY)
    message(FATAL_ERROR
        "libcupti was not found for SC_CUDA_HOME=${SC_CUDA_HOME}")
endif()

if(NOT SC_CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
        "CUDA Driver library libcuda.so was not found")
endif()

add_library(SuperColliderRuntime SHARED
    runtime/sc_runtime.cpp
)

target_include_directories(SuperColliderRuntime
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
        "${SC_CUDA_HOME}/include"
        "${SC_CUPTI_INCLUDE_DIR}"
)

target_link_libraries(SuperColliderRuntime
    PRIVATE
        "${SC_CUPTI_LIBRARY}"
        "${SC_CUDA_DRIVER_LIBRARY}"
        dl
        pthread
)

set_target_properties(SuperColliderRuntime PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "sc"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${SC_CUDA_HOME}/extras/CUPTI/lib64"
)