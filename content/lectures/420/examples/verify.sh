#!/bin/bash
# Verification script for CUDA examples on H200

set -e  # Exit on error

echo "========================================="
echo "CUDA Examples Verification Script"
echo "Target: NVIDIA H200 (Hopper SM 9.0)"
echo "========================================="
echo ""

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

echo "✓ CUDA compiler found: $(nvcc --version | grep release)"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found."
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Build all examples
echo "========================================="
echo "Building all examples..."
echo "========================================="
make clean
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All examples built successfully"
else
    echo ""
    echo "✗ Build failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Running examples..."
echo "========================================="
echo ""

# Run each example
for exe in 01_vector_add 02_vectorized_loads 03_matmul 04_coalescing 05_reduction; do
    echo "----------------------------------------"
    echo "Running: $exe"
    echo "----------------------------------------"
    if ./$exe; then
        echo "✓ $exe completed successfully"
    else
        echo "✗ $exe failed"
        exit 1
    fi
    echo ""
done

echo "========================================="
echo "All examples verified successfully!"
echo "========================================="
echo ""
echo "Expected performance on H200:"
echo "  - Vector Add: ~3 TB/s bandwidth"
echo "  - Vectorized Loads: ~4 TB/s bandwidth"
echo "  - Matmul Tiled: 500-800 GFLOP/s (FP32)"
echo "  - Coalescing: 2-3× speedup"
echo "  - Reduction: ~600 GB/s bandwidth"
echo ""
echo "To profile with Nsight Compute:"
echo "  ncu --set full ./01_vector_add"
echo "  ncu --set roofline ./03_matmul"
