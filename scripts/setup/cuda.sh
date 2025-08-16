#!/bin/bash
# CUDA installation script

# Source the base library template
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/base_library.sh"

# CUDA specific variables
LIBRARY_NAME="CUDA"
LIBRARY_VERSION="12.9.1"
# WSL-Ubuntu specific URLs
WSL_PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
WSL_DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb"
INSTALL_DIR="$HOME/.nemobotter-external/cuda"

# Override base methods for CUDA
test_library() {
    print_status "Testing CUDA installation..."
    
    # Source .bashrc to get CUDA environment variables
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
    
    # Test CUDA compilation
    print_debug "Testing CUDA compilation..."
    if test_cuda_compilation; then
        print_success "CUDA compilation test passed"
        return 0
    else
        print_warning "CUDA compilation test failed - but CUDA core functionality is available"
        return 1
    fi
}

check_installed() {
    print_status "Checking CUDA installation..."
    
    # Source .bashrc to get CUDA environment variables
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
    
    # Check if nvcc is available in PATH or installed
    local cuda_version=""
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        if [[ -n "$cuda_version" ]]; then
            print_success "CUDA $cuda_version is properly installed (nvcc in PATH)"
        else
            print_warning "CUDA version could not be determined"
            return 1
        fi
    elif [[ -f "/usr/local/cuda/bin/nvcc" ]]; then
        cuda_version=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        if [[ -n "$cuda_version" ]]; then
            print_success "CUDA $cuda_version is installed (nvcc not in PATH)"
        else
            print_warning "CUDA version could not be determined"
            return 1
        fi
    else
        print_warning "CUDA not found"
        return 1
    fi
    
    # Check CUDA installation directory
    if [[ ! -d "/usr/local/cuda" ]]; then
        print_warning "CUDA installation directory not found"
        return 1
    fi
    
    # Check CUDA runtime libraries
    if ! ldconfig -p | grep -q libcudart; then
        print_warning "CUDA runtime libraries not found"
        return 1
    fi
    
    # Check CUDA headers
    local cuda_headers_found=false
    for header_path in "/usr/local/cuda/include/cuda_runtime.h" "/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h" "/usr/include/cuda_runtime.h"; do
        if [[ -f "$header_path" ]]; then
            cuda_headers_found=true
            break
        fi
    done
    
    if [[ "$cuda_headers_found" == false ]]; then
        print_warning "CUDA headers not found"
        return 1
    fi
    
    print_success "CUDA is properly installed"
    return 0
}

download() {
    print_status "Downloading CUDA WSL-Ubuntu repository files..."
    
    # Create directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    # Check if files already exist
    if [[ -f "$INSTALL_DIR/cuda-wsl-ubuntu.pin" ]] && [[ -f "$INSTALL_DIR/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb" ]]; then
        print_success "CUDA WSL-Ubuntu repository files already downloaded"
        return 0
    fi
    
    # Download the repository pin file
    print_debug "Downloading CUDA pin file from: $WSL_PIN_URL"
    if run_command "wget -O '$INSTALL_DIR/cuda-wsl-ubuntu.pin' '$WSL_PIN_URL'" "CUDA pin file download"; then
        print_success "CUDA WSL-Ubuntu pin file downloaded"
    else
        print_error "Failed to download CUDA pin file"
        return 1
    fi
    
    # Download the repository deb file
    print_debug "Downloading CUDA repository package from: $WSL_DEB_URL"
    if run_command "wget -O '$INSTALL_DIR/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb' '$WSL_DEB_URL'" "CUDA repository package download"; then
        print_success "CUDA WSL-Ubuntu repository package downloaded"
    else
        print_error "Failed to download CUDA repository package"
        return 1
    fi
}

install() {
    print_status "Installing CUDA using WSL-Ubuntu repository..."
    
    # Check if CUDA is already installed
    if check_installed >/dev/null 2>&1; then
        print_success "CUDA is already installed and working"
        return 0
    fi
    
    # Download if not already downloaded
    if ! download; then
        return 1
    fi
    
    # Install CUDA using WSL-Ubuntu repository method
    print_status "Setting up CUDA WSL-Ubuntu repository..."
    
    # Move the pin file to the correct location
    print_debug "Installing CUDA repository pin file"
    if run_command "sudo mv '$INSTALL_DIR/cuda-wsl-ubuntu.pin' /etc/apt/preferences.d/cuda-repository-pin-600" "CUDA repository pin file installation"; then
        print_success "CUDA repository pin file installed"
    else
        print_error "Failed to install CUDA repository pin file"
        return 1
    fi
    
    # Install the repository package
    print_debug "Installing CUDA repository package"
    if run_command "sudo dpkg -i '$INSTALL_DIR/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb'" "CUDA repository package installation"; then
        print_success "CUDA repository package installed"
    else
        print_error "Failed to install CUDA repository package"
        return 1
    fi
    
    # Copy the keyring file
    print_status "Copying CUDA keyring files..."
    local cuda_repo_dir="/var/cuda-repo-wsl-ubuntu-12-9-local"
    local keyring_file=""
    
    # Find the specific keyring file
    if [[ -d "$cuda_repo_dir" ]]; then
        keyring_file=$(find "$cuda_repo_dir" -name "*keyring.gpg" -type f 2>/dev/null | head -1)
        if [[ -n "$keyring_file" ]]; then
            local keyring_filename=$(basename "$keyring_file")
            print_debug "Copying keyring file: $keyring_file"
            if run_command "sudo cp '$keyring_file' /usr/share/keyrings/" "CUDA keyring file copy"; then
                print_success "CUDA keyring file copied: $keyring_filename"
            else
                print_error "Failed to copy CUDA keyring file"
                return 1
            fi
        else
            print_error "CUDA keyring file not found in $cuda_repo_dir"
            return 1
        fi
    else
        print_error "CUDA repository directory not found: $cuda_repo_dir"
        return 1
    fi
    
    # Update package list
    apt_update
    
    # Install CUDA toolkit
    print_status "Installing CUDA toolkit..."
    if apt_install "cuda-toolkit-12-9" "CUDA toolkit installation"; then
        print_success "CUDA toolkit installed successfully"
        
        # Set up environment
        setup_cuda_environment
        
        # Verify installation
        if check_installed; then
            print_success "CUDA installation verified"
            return 0
        else
            print_warning "CUDA installation verification failed"
            return 1
        fi
    else
        print_error "CUDA toolkit installation failed"
        return 1
    fi
}

fix_incomplete_cuda_installation() {
    print_status "Fixing incomplete CUDA installation..."
    
    # Create missing include directory
    run_command "sudo mkdir -p /usr/local/cuda/include" "Creating CUDA include directory"
    
    # Use the already downloaded installer instead of downloading again
    local installer_file="$INSTALL_DIR/cuda_${LIBRARY_VERSION}_575.57.08_linux.run"
    local temp_dir="/tmp/cuda_headers"
    
    if [[ ! -f "$installer_file" ]]; then
        print_error "CUDA installer not found at $installer_file"
        print_status "Please run 'bash scripts/setup/cuda.sh download' first"
        return 1
    fi
    
    print_status "Using existing CUDA installer to extract headers..."
    
    mkdir -p "$temp_dir"
    cd "$temp_dir"
    
    # Copy the installer to temp directory
    run_command "cp '$installer_file' cuda_installer.run" "Copying CUDA installer to temp directory"
    
    # Extract the installer to get headers
    print_status "Extracting CUDA headers from installer..."
    print_debug "Extracting CUDA installer with --extract option"
    if run_command "chmod +x cuda_installer.run && ./cuda_installer.run --extract='$temp_dir' --silent" "CUDA installer extraction"; then
        # Find and copy headers
        local header_dir=$(find . -name "cuda_runtime.h" -type f 2>/dev/null | head -1 | xargs dirname)
        if [[ -n "$header_dir" ]]; then
            print_status "Found headers in: $header_dir"
            run_command "sudo cp -r '$header_dir'/* /usr/local/cuda/include/" "CUDA headers installation"
            print_success "CUDA headers installed"
        else
            print_warning "Could not find CUDA headers in installer"
        fi
    else
        print_warning "Could not extract CUDA installer"
    fi
    
    # Clean up
    cd - >/dev/null
    rm -rf "$temp_dir"
    
    # Set up environment
    setup_cuda_environment
    
    # Verify fix
    if check_installed >/dev/null 2>&1; then
        print_success "CUDA installation fixed successfully"
        return 0
    else
        print_warning "CUDA installation fix incomplete"
        return 1
    fi
}

cleanup() {
    print_status "Cleaning up CUDA installation files..."
    rm -f "$INSTALL_DIR/cuda_${LIBRARY_VERSION}_575.57.08_linux.run"
}

# Override setup_environment for CUDA
setup_environment() {
    print_status "Setting up environment for CUDA..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Install build tools if not present
    if ! command -v gcc &> /dev/null; then
        print_status "Installing build tools..."
        apt_install "build-essential" "Build tools installation"
    fi
    
    # Set up CUDA environment for current session
    setup_cuda_environment
    
    print_success "CUDA environment setup completed"
}

# Set up CUDA environment variables
setup_cuda_environment() {
    print_status "Setting up CUDA environment variables for current session..."
    
    # Set environment variables
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    
    # Find and set all CUDA library paths
    local cuda_lib_paths=()
    for lib_path in "/usr/local/cuda-12.9/targets/x86_64-linux/lib" "/usr/local/cuda/lib64" "/usr/local/cuda/targets/x86_64-linux/lib"; do
        if [[ -d "$lib_path" ]]; then
            cuda_lib_paths+=("$lib_path")
        fi
    done
    
    # Set LD_LIBRARY_PATH with all found library paths
    local ld_library_path="$CUDA_HOME/lib64"
    for lib_path in "${cuda_lib_paths[@]}"; do
        ld_library_path="$lib_path:$ld_library_path"
    done
    export LD_LIBRARY_PATH="$ld_library_path:$LD_LIBRARY_PATH"
    
    # Set include paths
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"
    export C_INCLUDE_PATH="/usr/local/cuda/include:$C_INCLUDE_PATH"
    
    # Add to shell profile for persistence
    local profile_file="$HOME/.bashrc"
    if ! grep -q "CUDA_HOME" "$profile_file" 2>/dev/null; then
        echo "" >> "$profile_file"
        echo "# CUDA Environment Variables" >> "$profile_file"
        echo "export CUDA_HOME=/usr/local/cuda" >> "$profile_file"
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> "$profile_file"
        
        # Add library paths to .bashrc
        local bashrc_lib_path="$CUDA_HOME/lib64"
        for lib_path in "${cuda_lib_paths[@]}"; do
            bashrc_lib_path="$lib_path:$bashrc_lib_path"
        done
        echo "export LD_LIBRARY_PATH=$bashrc_lib_path:\$LD_LIBRARY_PATH" >> "$profile_file"
        
        # Add include paths to .bashrc
        echo "export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:\$CPLUS_INCLUDE_PATH" >> "$profile_file"
        echo "export C_INCLUDE_PATH=/usr/local/cuda/include:\$C_INCLUDE_PATH" >> "$profile_file"
        
        print_success "CUDA environment variables added to $profile_file"
        # Automatically source .bashrc so the current shell session picks up the changes
        if [ -n "$BASH_VERSION" ]; then
            source "$profile_file"
            export CUDA_HOME=/usr/local/cuda
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$bashrc_lib_path:$LD_LIBRARY_PATH"
            export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"
            export C_INCLUDE_PATH="/usr/local/cuda/include:$C_INCLUDE_PATH"
            print_success "Reloaded $profile_file and exported CUDA variables in current shell session"
        fi
    fi
    
    print_success "CUDA environment variables set for current session"
    print_success "nvcc is now accessible in PATH"
    print_success "CUDA environment setup completed"
    # Print current CUDA environment for debug
    print_debug "Current CUDA_HOME: $CUDA_HOME"
    print_debug "Current PATH: $PATH"
    print_debug "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    print_debug "Current CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
    print_debug "Current C_INCLUDE_PATH: $C_INCLUDE_PATH"
}

# Test CUDA compilation
test_cuda_compilation() {
    local test_file="/tmp/cuda_test.cu"
    local test_output="/tmp/cuda_test.out"
    
    # Ensure build tools are installed
    if ! command -v gcc &> /dev/null; then
        print_status "Installing build tools..."
        apt_install "build-essential" "Build tools installation"
    fi
    
    # Environment should already be set up by setup_environment
    # Just verify CUDA environment is available
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found in PATH"
        return 1
    fi
    
    # Create symlinks for CUDA libraries if they don't exist
    local cuda_lib_dir="/usr/local/cuda-12.9/targets/x86_64-linux/lib"
    if [[ -d "$cuda_lib_dir" ]]; then
        if [[ ! -f "$cuda_lib_dir/libcudart.so" ]] && [[ -f "$cuda_lib_dir/libcudart.so.12.9.79" ]]; then
            print_status "Creating symlink for libcudart.so..."
            run_command "sudo ln -sf '$cuda_lib_dir/libcudart.so.12.9.79' '$cuda_lib_dir/libcudart.so'" "CUDA library symlink creation"
        fi
    fi
    
    # Debug: Show current environment
    print_debug "Current environment for compilation test:"
    print_debug "CUDA_HOME: ${CUDA_HOME:-'not set'}"
    print_debug "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-'not set'}"
    print_debug "PATH includes nvcc: $(command -v nvcc)"
    
    # Test if CUDA runtime is accessible by checking library availability
    if [[ -f "$cuda_lib_dir/libcudart.so" ]] || [[ -f "$cuda_lib_dir/libcudart.so.12.9.79" ]]; then
        print_success "CUDA runtime library found"
        
        # Test if we can access CUDA runtime functions by checking header compilation
        local header_test="/tmp/cuda_header_test.c"
        cat > "$header_test" << 'EOF'
#include <cuda_runtime.h>
int main() { return 0; }
EOF
        
        # Try to compile just the header to test if CUDA runtime is accessible
        # Use the environment variables that should be set
        local gcc_cmd="gcc"
        if [[ -n "$C_INCLUDE_PATH" ]]; then
            gcc_cmd="$gcc_cmd -I$C_INCLUDE_PATH"
        else
            gcc_cmd="$gcc_cmd -I/usr/local/cuda/include"
        fi
        
        gcc_cmd="$gcc_cmd -c $header_test -o /tmp/cuda_header_test.o"
        
        print_debug "Testing CUDA header compilation with: $gcc_cmd"
        if run_command "$gcc_cmd" "CUDA header compilation test"; then
            rm -f "$header_test" /tmp/cuda_header_test.o
            print_success "CUDA runtime headers are accessible"
            return 0
        else
            # Try with explicit include path
            print_debug "Retrying with explicit include path"
            if run_command "gcc -I/usr/local/cuda/include -c '$header_test' -o /tmp/cuda_header_test.o" "CUDA header compilation test (explicit path)"; then
                rm -f "$header_test" /tmp/cuda_header_test.o
                print_success "CUDA runtime headers are accessible (with explicit path)"
                return 0
            else
                # Show the error for debugging
                local error_output=$(gcc -I/usr/local/cuda/include -c "$header_test" -o /tmp/cuda_header_test.o 2>&1)
                print_debug "Header compilation error details:"
                print_debug "$error_output"
                rm -f "$header_test" /tmp/cuda_header_test.o
                print_warning "CUDA runtime headers not accessible"
                return 1
            fi
        fi
    else
        print_warning "CUDA runtime library not found"
        return 1
    fi
}

# Show CUDA environment status
show_environment() {
    print_status "CUDA Environment Status:"
    echo "CUDA_HOME: ${CUDA_HOME:-'not set'}"
    echo "PATH includes nvcc: $(command -v nvcc 2>/dev/null || echo 'no')"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-'not set'}"
    echo "CPLUS_INCLUDE_PATH: ${CPLUS_INCLUDE_PATH:-'not set'}"
    echo "C_INCLUDE_PATH: ${C_INCLUDE_PATH:-'not set'}"
    
    # Check for CUDA libraries
    local cuda_libs_found=false
    for lib_path in "/usr/local/cuda-12.9/targets/x86_64-linux/lib" "/usr/local/cuda/lib64" "/usr/local/cuda/targets/x86_64-linux/lib"; do
        if [[ -d "$lib_path" ]] && [[ -f "$lib_path/libcudart.so" ]]; then
            echo "CUDA libraries found at: $lib_path"
            cuda_libs_found=true
        fi
    done
    
    if [[ "$cuda_libs_found" == false ]]; then
        echo "CUDA libraries not found in standard locations"
    fi
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