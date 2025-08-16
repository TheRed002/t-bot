#!/bin/bash
# Test script to verify CUDA detection

echo "=== CUDA Detection Test ==="

# Test 1: Check if nvcc is in PATH
echo "1. Checking nvcc in PATH..."
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc found in PATH"
    nvcc --version
else
    echo "❌ nvcc not found in PATH"
fi

# Test 2: Check common CUDA locations
echo -e "\n2. Checking common CUDA locations..."
cuda_locations=(
    "/usr/local/cuda/bin/nvcc"
    "/usr/bin/nvcc"
    "/opt/cuda/bin/nvcc"
)

for location in "${cuda_locations[@]}"; do
    if [[ -f "$location" ]] && [[ -x "$location" ]]; then
        echo "✅ CUDA found at: $location"
        "$location" --version
    else
        echo "❌ CUDA not found at: $location"
    fi
done

# Test 3: Check CUDA installation directory
echo -e "\n3. Checking CUDA installation directory..."
if [[ -d "/usr/local/cuda" ]]; then
    echo "✅ CUDA installation directory found: /usr/local/cuda"
    ls -la /usr/local/cuda/bin/ | head -5
else
    echo "❌ CUDA installation directory not found"
fi

# Test 4: Check environment variables
echo -e "\n4. Checking CUDA environment variables..."
if [[ -n "$CUDA_HOME" ]]; then
    echo "✅ CUDA_HOME is set: $CUDA_HOME"
else
    echo "❌ CUDA_HOME not set"
fi

# Test 5: Check CUDA libraries
echo -e "\n5. Checking CUDA runtime libraries..."
if ldconfig -p | grep -q libcudart; then
    echo "✅ CUDA runtime libraries found"
    ldconfig -p | grep libcudart
else
    echo "❌ CUDA runtime libraries not found"
fi

# Test 6: Check PATH for CUDA
echo -e "\n6. Checking PATH for CUDA..."
echo "Current PATH: $PATH"
if echo "$PATH" | grep -q cuda; then
    echo "✅ CUDA found in PATH"
else
    echo "❌ CUDA not found in PATH"
fi

echo -e "\n=== Test Complete ===" 