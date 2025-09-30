# Makefile for GPU Programming Examples

NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3 -Xcompiler -Wall
ARCH = -arch=sm_90  # Hopper (H100/H200) - Default for this setup

# Detect GPU architecture automatically if needed
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
ifneq ($(CUDA_ARCH),)
    ARCH = -arch=sm_$(CUDA_ARCH)
endif

TARGETS = 01_vector_add 02_vectorized_loads 03_matmul 04_coalescing 05_reduction
CUTE_TARGETS = 06_cute_basics 07_cute_swizzling 08_cute_tiling 09_cute_paged_attention

# CuTe examples require CUTLASS 3.x+ headers
CUTLASS_DIR ?= $(HOME)/cutlass
CUTE_INCLUDES = -I$(CUTLASS_DIR)/include
CUTE_FLAGS = $(NVCC_FLAGS) $(CUTE_INCLUDES) -DCUTE_SM_90_ENABLED

all: $(TARGETS) cute_check

# Check if CUTLASS is available for CuTe examples
cute_check:
	@if [ -d "$(CUTLASS_DIR)" ]; then \
		echo "CUTLASS found at $(CUTLASS_DIR)"; \
		echo "Building CuTe examples..."; \
		$(MAKE) cute; \
	else \
		echo "CUTLASS not found at $(CUTLASS_DIR)"; \
		echo "Set CUTLASS_DIR to build CuTe examples"; \
		echo "  git clone https://github.com/NVIDIA/cutlass.git $(HOME)/cutlass"; \
	fi

cute: $(CUTE_TARGETS)

01_vector_add: 01_vector_add.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<

02_vectorized_loads: 02_vectorized_loads.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<

03_matmul: 03_matmul.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<

04_coalescing: 04_coalescing.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<

05_reduction: 05_reduction.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $<

# CuTe DSL examples
06_cute_basics: 06_cute_basics.cu common.cuh
	$(NVCC) $(CUTE_FLAGS) $(ARCH) -o $@ $<

07_cute_swizzling: 07_cute_swizzling.cu common.cuh
	$(NVCC) $(CUTE_FLAGS) $(ARCH) -o $@ $<

08_cute_tiling: 08_cute_tiling.cu common.cuh
	$(NVCC) $(CUTE_FLAGS) $(ARCH) -o $@ $<

09_cute_paged_attention: 09_cute_paged_attention.cu common.cuh
	$(NVCC) $(CUTE_FLAGS) $(ARCH) -o $@ $<

# Python example (no compilation needed)
python: 10_cute_python_paged_attention.py
	@echo "Python example ready. Run with:"
	@echo "  python 10_cute_python_paged_attention.py"

clean:
	rm -f $(TARGETS) $(CUTE_TARGETS) *.o

test: all
	@echo "Running all basic examples..."
	@echo ""
	@echo "================================"
	@./01_vector_add
	@echo "================================"
	@./02_vectorized_loads
	@echo "================================"
	@./03_matmul
	@echo "================================"
	@./04_coalescing
	@echo "================================"
	@./05_reduction

test_cute: cute
	@echo "Running CuTe DSL examples..."
	@echo ""
	@echo "================================"
	@./06_cute_basics
	@echo "================================"
	@./07_cute_swizzling
	@echo "================================"
	@./08_cute_tiling
	@echo "================================"
	@./09_cute_paged_attention

test_python:
	@echo "Running Python CuTe DSL example..."
	python 10_cute_python_paged_attention.py

test_all: test test_cute test_python

.PHONY: all clean test cute cute_check test_cute test_python test_all python
