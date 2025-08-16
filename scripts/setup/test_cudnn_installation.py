#!/usr/bin/env python3
"""
Test script to verify cuDNN installation and show installed libraries.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_cudnn_libraries():
    """Check for cuDNN libraries in the system."""
    print("🔍 Checking cuDNN libraries...")
    
    # Check if cuDNN libraries are available
    success, output, error = run_command("ldconfig -p | grep cudnn")
    
    if success and output.strip():
        print("✅ cuDNN libraries found:")
        for line in output.strip().split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("❌ No cuDNN libraries found")
        return False
    
    return True

def check_cudnn_package():
    """Check if cuDNN package is installed via apt."""
    print("\n📦 Checking cuDNN package installation...")
    
    success, output, error = run_command("dpkg -l | grep cudnn")
    
    if success and output.strip():
        print("✅ cuDNN packages installed:")
        for line in output.strip().split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("❌ No cuDNN packages found via dpkg")
        return False
    
    return True

def check_cudnn_files():
    """Check for cuDNN files in common locations."""
    print("\n📁 Checking cuDNN files in common locations...")
    
    cudnn_locations = [
        "/usr/local/cuda/include/cudnn.h",
        "/usr/include/cudnn.h",
        "/usr/local/cuda/lib64/libcudnn.so",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so"
    ]
    
    found_files = []
    for location in cudnn_locations:
        if os.path.exists(location):
            found_files.append(location)
            print(f"✅ Found: {location}")
    
    if not found_files:
        print("❌ No cuDNN files found in common locations")
        return False
    
    return True

def check_cudnn_version():
    """Check cuDNN version if possible."""
    print("\n🔢 Checking cuDNN version...")
    
    # Try to get cuDNN version from header file (check multiple locations)
    cudnn_header_locations = [
        "/usr/local/cuda/include/cudnn.h",
        "/usr/include/cudnn.h",
        "/usr/local/include/cudnn.h",
        "/usr/include/x86_64-linux-gnu/cudnn.h"  # Ubuntu/Debian standard location
    ]
    
    cudnn_header = None
    for location in cudnn_header_locations:
        if os.path.exists(location):
            cudnn_header = location
            break
    
    if cudnn_header:
        try:
            with open(cudnn_header, 'r') as f:
                content = f.read()
                # Look for version defines
                if "CUDNN_MAJOR" in content:
                    print("✅ cuDNN header file found with version information")
                    # Extract version info if possible
                    lines = content.split('\n')
                    for line in lines:
                        if "CUDNN_MAJOR" in line or "CUDNN_MINOR" in line or "CUDNN_PATCHLEVEL" in line:
                            print(f"   {line.strip()}")
                else:
                    print("⚠️  cuDNN header file found but version info not easily extractable")
        except Exception as e:
            print(f"⚠️  Could not read cuDNN header file: {e}")
    else:
        print("❌ cuDNN header file not found")
        print("💡 To install cuDNN development headers, run:")
        print("   sudo apt-get install libcudnn9-dev-cuda-12")

def check_cuda_cudnn_compatibility():
    """Check CUDA and cuDNN compatibility."""
    print("\n🔗 Checking CUDA and cuDNN compatibility...")
    
    # Check CUDA version
    success, output, error = run_command("nvcc --version")
    if success:
        print("✅ CUDA compiler found:")
        for line in output.strip().split('\n'):
            if "release" in line:
                print(f"   {line.strip()}")
    else:
        print("❌ CUDA compiler not found")
    
    # Check if cuDNN is compatible with CUDA
    cuda_lib = "/usr/local/cuda/lib64"
    if os.path.exists(cuda_lib):
        success, output, error = run_command(f"ls {cuda_lib} | grep cudnn")
        if success and output.strip():
            print("✅ cuDNN libraries found in CUDA directory:")
            for line in output.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")

def check_python_cudnn():
    """Check if Python can access cuDNN."""
    print("\n🐍 Checking Python cuDNN access...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            
            # Try to access cuDNN
            if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                print("✅ PyTorch cuDNN is available")
                print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            else:
                print("❌ PyTorch cuDNN is not available")
        else:
            print("❌ PyTorch CUDA is not available")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            print("✅ TensorFlow GPU is available")
            print(f"   TensorFlow version: {tf.__version__}")
            
            # Check cuDNN in TensorFlow
            if hasattr(tf, 'keras') and hasattr(tf.keras.backend, 'image_data_format'):
                print("✅ TensorFlow cuDNN support available")
            else:
                print("⚠️  TensorFlow cuDNN support not confirmed")
        else:
            print("❌ TensorFlow GPU is not available")
    except ImportError:
        print("⚠️  TensorFlow not installed")

def main():
    """Main function to run all cuDNN checks."""
    print("=" * 60)
    print("🧠 cuDNN Installation Verification")
    print("=" * 60)
    
    # Run all checks
    checks = [
        check_cudnn_libraries,
        check_cudnn_package,
        check_cudnn_files,
        check_cudnn_version,
        check_cuda_cudnn_compatibility,
        check_python_cudnn
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Error during check: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks")
    
    if passed >= 4:
        print("✅ cuDNN installation appears to be successful!")
    elif passed >= 2:
        print("⚠️  cuDNN installation is partially successful")
    else:
        print("❌ cuDNN installation may have failed")
    
    print("\n💡 Next steps:")
    print("   - If installation failed, check the logs above")
    print("   - Restart your terminal and run: source ~/.bashrc")
    print("   - Test with: python scripts/test_gpu_dev.py")

if __name__ == "__main__":
    main() 