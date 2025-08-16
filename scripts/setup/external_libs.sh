#!/bin/bash
# Master script to install all external libraries

# Source the base library template
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/base_library.sh"

# Library scripts
LIBRARY_SCRIPTS=(
    "cuda.sh"
    "cudnn.sh"
    "talib.sh"
    "lightgbm.sh"
)

# Override base methods for master script
test_libraries() {
    print_status "Testing all external libraries..."
    
    local all_tests_passed=true
    local failed_tests=()
    
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        local lib_name=$(basename "$script" .sh)
        print_status "Testing $lib_name..."
        
        # Use absolute path to the script
        local script_path="$SCRIPT_DIR/$script"
        if [[ "$DEBUG_FLAG" == "true" ]]; then
            print_debug "Testing $lib_name: bash \"$script_path\" test"
        fi
        
        # Run the test command and capture the exit code
        if bash "$script_path" test; then
            print_success "$lib_name tests passed"
        else
            print_warning "$lib_name tests failed"
            failed_tests+=("$lib_name")
            all_tests_passed=false
        fi
    done
    
    if [[ "$all_tests_passed" == "true" ]]; then
        print_success "All external library tests passed"
        return 0
    else
        print_warning "Some external library tests failed: ${failed_tests[*]}"
        return 1
    fi
}

check_installed() {
    print_status "Checking all external libraries..."
    
    local all_installed=true
    local failed_libraries=()
    
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        local lib_name=$(basename "$script" .sh)
        print_status "Checking $lib_name..."
        
        # Use absolute path to the script
        local script_path="$SCRIPT_DIR/$script"
        if [[ "$DEBUG_FLAG" == "true" ]]; then
            print_debug "Running: bash \"$script_path\" check"
        fi
        
        # Run the check command and capture the exit code
        if bash "$script_path" check; then
            print_success "$lib_name is properly installed"
        else
            print_warning "$lib_name is not properly installed"
            failed_libraries+=("$lib_name")
            all_installed=false
        fi
    done
    
    if [[ "$all_installed" == "true" ]]; then
        print_success "All external libraries are installed"
        return 0
    else
        print_warning "Some external libraries are not installed: ${failed_libraries[*]}"
        return 1
    fi
}

download() {
    print_status "Downloading all external libraries..."
    
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        local lib_name=$(basename "$script" .sh)
        print_status "Downloading $lib_name..."
        
        # Only download if not already installed
        # Use absolute path to the script
        local script_path="$SCRIPT_DIR/$script"
        
        if bash "$script_path" check; then
            print_success "$lib_name is already installed - skipping download"
            continue
        fi
        
        if [[ "$DEBUG_FLAG" == "true" ]]; then
            print_debug "Downloading $lib_name: bash \"$script_path\" download"
        fi
        
        if ! bash "$script_path" download; then
            print_warning "Failed to download $lib_name"
        fi
    done
}

install() {
    print_status "Installing all external libraries..."
    
    # Install libraries in dependency order
    local install_order=(
        "cuda.sh"      # CUDA first (dependency for others)
        "cudnn.sh"     # cuDNN second (depends on CUDA)
        "talib.sh"     # TA-Lib third (independent)
        "lightgbm.sh"  # LightGBM last (depends on CUDA)
    )
    
    local failed_libraries=()
    local skipped_libraries=()
    
    for script in "${install_order[@]}"; do
        local lib_name=$(basename "$script" .sh)
        print_status "Installing $lib_name..."
        
        # Use relative path from current directory
        
        # Check if already properly installed
        local script_path="$SCRIPT_DIR/$script"
        if bash "$script_path" check >/dev/null 2>&1; then
            print_success "$lib_name is already properly installed - skipping"
            skipped_libraries+=("$lib_name")
            continue
        fi
        
        # Try to install the library
        if [[ "$DEBUG_FLAG" == "true" ]]; then
            print_debug "Installing $lib_name: bash \"$script_path\" install"
        fi
        
        if bash "$script_path" install; then
            print_success "$lib_name installed successfully"
        else
            print_warning "Failed to install $lib_name"
            failed_libraries+=("$lib_name")
        fi
    done
    
    # Report results
    if [[ ${#failed_libraries[@]} -eq 0 ]]; then
        if [[ ${#skipped_libraries[@]} -gt 0 ]]; then
            print_success "All external libraries are ready (${#skipped_libraries[@]} already installed, ${#install_order[@]} total)"
        else
            print_success "All external libraries installed successfully"
        fi
        return 0
    else
        print_warning "Some libraries failed to install: ${failed_libraries[*]}"
        print_status "You can try installing them individually:"
        for lib in "${failed_libraries[@]}"; do
            print_status "  bash scripts/setup/${lib}.sh install"
        done
        return 1
    fi
}

cleanup() {
    print_status "Cleaning up all external libraries..."
    
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        print_status "Cleaning up $script..."
        bash "$SCRIPT_DIR/$script" cleanup
    done
}

# Override setup_environment for master script
setup_environment() {
    print_status "Setting up environment for all external libraries..."
    
    # Create installation directories
    mkdir -p "$HOME/.nemobotter-external"
    mkdir -p "$HOME/.nemobotter-cuda-tools"
    
    # Only install common dependencies if not already done
    if ! command -v wget &> /dev/null || ! command -v gcc &> /dev/null; then
        print_status "Installing common dependencies..."
        apt_install "wget build-essential" "Common dependencies installation"
    fi
    
    print_success "Master environment setup completed"
}

# Show status of all libraries
show_status() {
    print_status "External Libraries Status:"
    echo "=================================="
    
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        local lib_name=$(basename "$script" .sh)
        echo -n "$lib_name: "
        
        # Use absolute path to the script
        local script_path="$SCRIPT_DIR/$script"
        if [[ "$DEBUG_FLAG" == "true" ]]; then
            print_debug "Checking $lib_name: bash \"$script_path\" check"
        fi
        
        # Run the check command and capture the exit code
        if bash "$script_path" check; then
            echo -e "${GREEN}✓ Installed${NC}"
        else
            echo -e "${RED}✗ Not Installed${NC}"
        fi
    done
}

# Override show_usage for master script
show_usage() {
    echo "Usage: $0 [check|download|install|cleanup|status|test] [--debug]"
    echo "  check    - Check if all libraries are installed"
    echo "  download - Download all library files"
    echo "  install  - Install all libraries (includes download and install)"
    echo "  cleanup  - Clean up temporary files for all libraries"
    echo "  status   - Show status of all libraries"
    echo "  test     - Run comprehensive tests on all libraries"
    echo "  --debug  - Enable debug mode with detailed logging"
    echo ""
    echo "Individual library commands:"
    for script in "${LIBRARY_SCRIPTS[@]}"; do
        local lib_name=$(basename "$script" .sh)
        echo "  $lib_name [check|download|install|cleanup|test] [--debug]"
    done
    echo ""
    echo "Examples:"
    echo "  $0 install --debug          # Install all libraries with debug output"
    echo "  $0 cuda install --debug     # Install CUDA with debug output"
    echo "  $0 check                    # Check all libraries"
}

# Override main function for master script
main() {
    local command="${1:-install}"
    local debug_flag=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug)
                debug_flag=true
                shift
                ;;
            check|download|install|cleanup|status|test|help|-h|--help)
                command="$1"
                shift
                ;;
            *)
                # Check if it's an individual library command
                for script in "${LIBRARY_SCRIPTS[@]}"; do
                    local lib_name=$(basename "$script" .sh)
                    if [[ "$1" == "$lib_name" ]]; then
                        shift
                        # Pass debug flag to individual script
                        if [[ "$debug_flag" == "true" ]]; then
                            bash "$SCRIPT_DIR/$script" "$@" --debug
                        else
                            bash "$SCRIPT_DIR/$script" "$@"
                        fi
                        exit $?
                    fi
                done
                
                print_error "Unknown command: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set debug flag for this script
    if [[ "$debug_flag" == "true" ]]; then
        DEBUG_FLAG=true
        print_debug "Debug mode enabled"
    fi
    
    case "$command" in
        "check")
            check_installed
            ;;
        "download")
            setup_environment
            download
            ;;
        "install")
            setup_environment
            install
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            show_status
            ;;
        "test")
            test_libraries
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
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