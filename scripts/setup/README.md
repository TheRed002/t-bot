# Setup Scripts

This directory contains setup scripts for external libraries and tools used by the NemoBotter trading system.

## Overview

The setup system provides a comprehensive way to install, check, and test external libraries with proper dependency management and clean output.

## Master Script: `external_libs.sh`

The main entry point for managing all external libraries:

```bash
./scripts/setup/external_libs.sh [command] [--debug]
```

### Available Commands

- **`check`** - Verify if libraries are installed (file existence only)
- **`test`** - Run comprehensive tests on all libraries
- **`download`** - Download library source files
- **`install`** - Install all libraries in dependency order
- **`cleanup`** - Clean up temporary files
- **`status`** - Show installation status of all libraries
- **`--debug`** - Enable detailed logging

### Examples

```bash
# Check if all libraries are installed
./scripts/setup/external_libs.sh check

# Run comprehensive tests on all libraries
./scripts/setup/external_libs.sh test

# Install all libraries with debug output
./scripts/setup/external_libs.sh install --debug

# Show status of all libraries
./scripts/setup/external_libs.sh status
```

## Individual Library Scripts

Each library has its own script with the same command structure:

```bash
./scripts/setup/[library].sh [command] [--debug]
```

### Available Libraries

#### 1. CUDA (`cuda.sh`)
- **Purpose**: NVIDIA CUDA toolkit and drivers for GPU acceleration
- **Dependencies**: None (installed first)
- **Commands**: `check`, `test`, `download`, `install`, `cleanup`
- **Test**: Compilation verification with CUDA headers

#### 2. cuDNN (`cudnn.sh`)
- **Purpose**: NVIDIA cuDNN library for deep learning primitives
- **Dependencies**: CUDA (must be installed first)
- **Commands**: `check`, `test`, `download`, `install`, `cleanup`
- **Test**: Official NVIDIA sample compilation and execution

#### 3. TA-Lib (`talib.sh`)
- **Purpose**: Technical Analysis Library for trading indicators
- **Dependencies**: None (independent)
- **Commands**: `check`, `test`, `download`, `install`, `cleanup`
- **Test**: Python module import and basic functionality

#### 4. LightGBM (`lightgbm.sh`)
- **Purpose**: Machine learning library with CUDA GPU support
- **Dependencies**: CUDA (must be installed first)
- **Commands**: `check`, `test`, `download`, `install`, `cleanup`
- **Test**: CUDA and CPU training verification

## Installation Order

Due to dependencies, libraries should be installed in this order:

1. **CUDA** (`cuda.sh`) - Base GPU support
2. **cuDNN** (`cudnn.sh`) - Deep learning primitives  
3. **TA-Lib** (`talib.sh`) - Technical analysis (independent)
4. **LightGBM** (`lightgbm.sh`) - Machine learning with GPU

## Command Details

### Check Command
- **Purpose**: Verify if libraries are installed by checking file existence
- **Speed**: Fast (no compilation or testing)
- **Use case**: Quick status verification
- **Example**: `./scripts/setup/external_libs.sh check`

### Test Command
- **Purpose**: Run comprehensive tests to verify functionality
- **Speed**: Slower (includes compilation and execution tests)
- **Use case**: Thorough verification after installation
- **Example**: `./scripts/setup/external_libs.sh test`

### Install Command
- **Purpose**: Download and install all libraries in dependency order
- **Speed**: Slowest (downloads and compiles everything)
- **Use case**: Initial setup or reinstallation
- **Example**: `./scripts/setup/external_libs.sh install`

## Output and Logging

### Normal Mode
- Clean, minimal output
- Only essential status messages
- Success/failure indicators

### Debug Mode (`--debug`)
- Detailed logging
- Step-by-step progress
- Error details and troubleshooting info

### Example Output

```bash
# Normal mode
[INFO] External Libraries Status:
==================================
cuda: ✓ Installed
cudnn: ✓ Installed
talib: ✓ Installed
lightgbm: ✓ Installed

# Debug mode
[DEBUG] Checking cuda: bash "/path/to/cuda.sh" check
[INFO] Checking CUDA installation...
[SUCCESS] CUDA 12.9 is installed (nvcc not in PATH)
[SUCCESS] CUDA is properly installed
```

## Test Scripts

Individual test scripts for specific verification:

### 1. CUDA Test (`test_cuda_detection.sh`)
```bash
./scripts/setup/test_cuda_detection.sh
```

### 2. cuDNN Test (`test_cudnn_installation.py`)
```bash
./scripts/setup/test_cudnn_installation.py
```

### 3. TA-Lib Test (`test_talib_installation.py`)
```bash
./scripts/setup/test_talib_installation.py
```

### 4. LightGBM Test (`test_lightgbm_gpu.py`)
```bash
./scripts/setup/test_lightgbm_gpu.py
```

## Troubleshooting

### Common Issues

#### CUDA Issues
- **"nvcc not found in PATH"**: Normal in WSL, CUDA still functional
- **"CUDA compilation test failed"**: Expected if nvcc not in PATH, core functionality available
- **Driver issues**: Update NVIDIA drivers to match CUDA version

#### cuDNN Issues
- **"cuDNN not found"**: Ensure CUDA is installed first
- **"FreeImage not found"**: Automatically installed during test
- **Version mismatch**: Check CUDA and cuDNN version compatibility

#### TA-Lib Issues
- **"TA-Lib header files not found"**: Check installation path
- **Compilation errors**: Install build dependencies: `sudo apt install build-essential`
- **Import errors**: Check if TA-Lib is installed in the virtual environment

#### LightGBM Issues
- **"CUDA Tree Learner was not enabled"**: Reinstall with CUDA support
- **"No CUDA device found"**: Check CUDA installation and GPU availability

### Debug Mode
Use `--debug` flag for detailed troubleshooting:
```bash
./scripts/setup/external_libs.sh test --debug
```

## Environment Requirements

- **Operating System**: Linux (WSL supported)
- **Virtual Environment**: All scripts use `~/.venv` for isolation
- **GPU Hardware**: CUDA support requires NVIDIA GPU
- **Build Tools**: C++ compiler and cmake for source compilation
- **Package Manager**: apt for system dependencies

## Notes

- **Idempotent**: Scripts are safe to run multiple times
- **Dependency Management**: Automatic installation order handling
- **Clean Output**: Minimal logging by default, detailed with `--debug`
- **WSL Optimized**: Designed for WSL environments with proper path handling
- **Error Recovery**: Graceful handling of partial failures
- **Space Handling**: Properly handles spaces in directory names

## Complete Setup

### Main Setup Script (`setup.sh`)
The main setup script that installs everything:

```bash
# Complete setup (recommended)
./scripts/setup/setup.sh

# Complete setup with debug output
./scripts/setup/setup.sh --debug
```

This script will:
1. Install system packages
2. Install external libraries (CUDA, cuDNN, TA-Lib, LightGBM)
3. Create virtual environment
4. Install Python packages
5. Test external libraries
6. Test research tools

### Individual Library Management

```bash
# Install all libraries
./scripts/setup/external_libs.sh install

# Check installation status
./scripts/setup/external_libs.sh status

# Run comprehensive tests
./scripts/setup/external_libs.sh test

# Individual library management
./scripts/setup/cuda.sh check
./scripts/setup/lightgbm.sh test --debug
``` 