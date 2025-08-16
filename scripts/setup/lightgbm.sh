#!/bin/bash
# LightGBM installation script with CUDA GPU support

# Source the base library template
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/base_library.sh"

# LightGBM specific variables
LIBRARY_NAME="LightGBM"
LIBRARY_VERSION="4.6.0"
DOWNLOAD_URL="https://github.com/microsoft/LightGBM"
INSTALL_DIR="$HOME/.nemobotter-external/lightgbm"

# Override base methods for LightGBM
test_library() {
    print_status "Testing LightGBM installation..."
    
    # Test CUDA support first, then CPU as fallback
    print_status "Testing LightGBM CUDA support..."
    
    # Method 1: Try CUDA support
    print_debug "Testing CUDA support..."
    if "$VENV_PATH/bin/python" -c "
import lightgbm as lgb
import numpy as np
try:
    # Create proper numpy arrays for LightGBM
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    data = lgb.Dataset(X, label=y)
    params = {'device': 'cuda', 'verbose': -1}
    bst = lgb.train(params, data, num_boost_round=1)
    print('LightGBM CUDA support verified successfully')
except Exception as e:
    print('LightGBM CUDA support verification failed:', str(e))
    print('Exception type:', type(e).__name__)
    exit(1)
" 2>&1 | tee /tmp/lightgbm_cuda_test.log >/dev/null; then
        if grep -q "LightGBM CUDA support verified successfully" /tmp/lightgbm_cuda_test.log; then
            print_success "LightGBM with CUDA support is installed and working"
            return 0
        else
            print_debug "CUDA test output:"
            cat /tmp/lightgbm_cuda_test.log
        fi
    fi
    
    # Method 3: Try CPU fallback
    print_debug "Testing CPU fallback..."
    if "$VENV_PATH/bin/python" -c "
import lightgbm as lgb
import numpy as np
try:
    # Create proper numpy arrays for LightGBM
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    data = lgb.Dataset(X, label=y)
    params = {'device': 'cpu'}
    bst = lgb.train(params, data, num_boost_round=1)
    print('LightGBM CPU support verified successfully')
except Exception as e:
    print('LightGBM CPU support verification failed:', str(e))
    print('Exception type:', type(e).__name__)
    exit(1)
" 2>&1 | tee /tmp/lightgbm_cpu_test.log >/dev/null; then
        if grep -q "LightGBM CPU support verified successfully" /tmp/lightgbm_cpu_test.log; then
            print_warning "LightGBM installed but only CPU support available"
            return 0
        else
            print_debug "CPU test output:"
            cat /tmp/lightgbm_cpu_test.log
        fi
    fi
    
    print_warning "LightGBM installed but GPU support not available"
    return 1
}

check_installed() {
    print_status "Checking LightGBM installation..."
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        print_warning "Virtual environment not found"
        return 1
    fi
    
    # Check if LightGBM is installed in virtual environment
    if ! "$VENV_PATH/bin/python" -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)" 2>/dev/null; then
        print_warning "LightGBM not installed in virtual environment"
        return 1
    fi
    
    print_success "LightGBM is installed and working"
    return 0
}

check_existing_build() {
    print_debug "Checking for existing LightGBM build..."
    
    # Check if LightGBM repo is cloned
    if [[ ! -d "$INSTALL_DIR/LightGBM" ]]; then
        print_debug "LightGBM repository not found"
        return 1
    fi
    
    # Check if lib_lightgbm.so exists (main library) - this is the essential file
    if [[ ! -f "$INSTALL_DIR/LightGBM/lib_lightgbm.so" ]]; then
        print_debug "lib_lightgbm.so not found"
        return 1
    fi
    
    # # Check if lightgbm executable exists
    # if [[ ! -f "$INSTALL_DIR/LightGBM/lightgbm" ]]; then
    #     print_debug "lightgbm executable not found"
    #     return 1
    # fi
    
    # Check if python-package directory exists with required files
    if [[ ! -d "$INSTALL_DIR/LightGBM/python-package" ]]; then
        print_debug "python-package directory not found"
        return 1
    fi
    
    # Check for required files in python-package
    local required_files=("README.rst" "lightgbm" "pyproject.toml")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$INSTALL_DIR/LightGBM/python-package/$file" ]] && [[ ! -d "$INSTALL_DIR/LightGBM/python-package/$file" ]]; then
            print_debug "Required file/directory not found: python-package/$file"
            return 1
        fi
    done
    
    print_debug "All required build artifacts found"
    return 0
}

download() {
    # print_status "Downloading LightGBM source from GitHub..."
    
    # # Create directory if it doesn't exist
    # mkdir -p "$INSTALL_DIR" 
    # cd "$INSTALL_DIR"
    
    # # Check if already downloaded
    # if [[ -d "LightGBM" ]]; then
    #     print_success "LightGBM source already downloaded"
    #     return 0
    # fi
    
    # # Clone LightGBM repository
    # print_debug "Cloning LightGBM from: $DOWNLOAD_URL"
    # if run_command "git clone --recursive '$DOWNLOAD_URL'" "LightGBM repository clone"; then
    #     print_success "LightGBM source downloaded successfully"
    #     return 0
    # else
    #     print_error "Failed to download LightGBM source"
    #     return 1
    # fi
    return 0
}

install() {
    print_status "Installing LightGBM with CUDA GPU support..."
    
    # Check if CUDA is available (should be installed by cuda.sh)
    local cuda_available=false
    
    # Check for CUDA installation
    if [[ -f "/usr/local/cuda/bin/nvcc" ]]; then
        cuda_available=true
        print_status "CUDA found at: /usr/local/cuda"
    fi
    
    if [[ "$cuda_available" == false ]]; then
        print_warning "CUDA not found"
        print_status "Please install CUDA first: ./scripts/setup/cuda.sh install"
        print_status "Trying CPU-only installation..."
        "$VENV_PATH/bin/pip" install lightgbm>=$LIBRARY_VERSION
        return 0
    fi
    
    # Uninstall existing LightGBM first to ensure clean installation
    print_status "Uninstalling existing LightGBM..."
    print_debug "Uninstalling existing LightGBM package"
    run_command "\"$VENV_PATH/bin/pip\" uninstall lightgbm -y" "LightGBM uninstall"
    
        # Method 1: Try pip installation with CUDA support (recommended approach)
    print_status "Installing LightGBM with CUDA support via pip..."
    
    # Install build dependencies
    print_status "Installing build dependencies..."
    apt_install "build-essential cmake git libboost-dev libboost-system-dev libboost-filesystem-dev" "Build dependencies installation"
    
    # Install LightGBM with CUDA support using pip (exact command from Stack Overflow)
    print_debug "Installing LightGBM with CUDA support via pip"
    print_debug "Clearing pip cache to force source compilation"
    run_command "\"$VENV_PATH/bin/pip\" cache purge" "Pip cache purge"
    if run_command "\"$VENV_PATH/bin/pip\" install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON 'lightgbm>=4.0.0'" "LightGBM CUDA pip installation"; then
        print_success "LightGBM with CUDA support installed via pip"
        return 0
    else
        print_warning "CUDA pip installation failed, trying CPU version"
    fi
    
    # Method 2: Try CPU-only pip installation (fallback)
    print_status "Trying CPU-only pip installation..."
    print_debug "Installing LightGBM CPU version via pip"
    if run_command "\"$VENV_PATH/bin/pip\" install lightgbm" "LightGBM CPU pip installation"; then
        print_success "LightGBM CPU version installed via pip"
        return 0
    fi
    
    # Method 3: Try with conda (if available)
    if command -v conda &> /dev/null; then
        print_status "Trying conda installation..."
        print_debug "Installing LightGBM via conda"
        if run_command "conda install -c conda-forge lightgbm -y" "LightGBM conda installation"; then
            print_success "LightGBM installed via conda"
            return 0
        fi
    fi
    
    # Method 4: Try conda-forge wheel with GPU support
    if command -v conda &> /dev/null; then
        print_status "Trying conda-forge LightGBM with GPU support..."
        print_debug "Installing LightGBM via conda-forge"
        if run_command "conda install -c conda-forge lightgbm -y" "LightGBM conda-forge installation"; then
            print_success "LightGBM conda-forge wheel installed"
            return 0
        fi
    fi
    
    # Method 5: Fallback to CPU version
    print_warning "CUDA GPU installation methods failed, installing CPU version"
    print_debug "Installing LightGBM CPU version via pip"
    run_command "\"$VENV_PATH/bin/pip\" install lightgbm>=$LIBRARY_VERSION" "LightGBM CPU installation"
    
    return 0
}

cleanup() {
    print_status "Cleaning up LightGBM installation files..."
    # Keep the source for future use, but clean build artifacts
    if [[ -d "$INSTALL_DIR/LightGBM" ]]; then
        print_debug "Cleaning LightGBM build artifacts"
        run_command "rm -rf '$INSTALL_DIR/LightGBM/build'" "LightGBM build cleanup"
    fi
}

# Override setup_environment for LightGBM
setup_environment() {
    print_status "Setting up environment for LightGBM..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Set environment variables for CUDA compilation
    if [[ -f "/usr/local/cuda" ]]; then
        export CUDA_HOME="/usr/local/cuda"
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        print_status "CUDA environment variables set"
    fi
    
    # Set LightGBM specific environment variables for CUDA
    export LIGHTGBM_USE_CUDA=1
    
    print_success "LightGBM environment setup completed"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Ensure main function is available from base library
    if declare -f main > /dev/null; then
        main "$@"
    else
        print_error "Main function not found. Please check base library sourcing."
        exit 1
    fi
fi 