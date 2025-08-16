#!/bin/bash

# NemoBotter Trading System - Complete Setup Script
# Creates virtual environment and installs all dependencies including external libraries

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [[ "$DEBUG_FLAG" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        exit 1
    fi
}

# Function to check system requirements
check_system() {
    print_status "Checking system requirements..."
    
    # Check if we're on Ubuntu/Debian
    if ! command_exists apt; then
        print_error "This script requires Ubuntu/Debian system with apt package manager"
        exit 1
    fi
    
    # Check Python version
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Found Python $python_version"
    
    # Validate project structure
    validate_project_structure
}

# Function to validate project structure
validate_project_structure() {
    print_status "Validating project structure..."
    
    print_status "Current working directory: $(pwd)"
    
    # Try multiple methods to find the project root
    local project_root=""
    local requirements_found=false
    
    # Method 1: Check if we're already in the project root
    if [[ -f "requirements/base.txt" ]] && [[ -f "requirements/development.txt" ]]; then
        project_root="$(pwd)"
        print_success "Found requirements files in current directory"
        requirements_found=true
    fi
    
    # Method 2: Try to find project root relative to script location
    if [[ "$requirements_found" == false ]]; then
        local script_path="${BASH_SOURCE[0]}"
        print_debug "Script path: $script_path"
        
        if [[ -n "$script_path" ]] && [[ -f "$script_path" ]]; then
            local script_dir="$(cd "$(dirname "$script_path")" && pwd 2>/dev/null)"
            if [[ -n "$script_dir" ]]; then
                print_debug "Script directory: $script_dir"
                local potential_root="$(cd "$script_dir/.." && pwd 2>/dev/null)"
                if [[ -n "$potential_root" ]] && [[ -f "$potential_root/requirements/base.txt" ]]; then
                    project_root="$potential_root"
                    print_success "Found requirements files in project root: $project_root"
                    requirements_found=true
                fi
            fi
        fi
    fi
    
    # Method 3: Search for project root from current directory
    if [[ "$requirements_found" == false ]]; then
        print_status "Searching for project root from current directory..."
        local current_dir="$(pwd)"
        local search_dir="$current_dir"
        
        # Search up to 5 levels up
        for i in {1..5}; do
            if [[ -f "$search_dir/requirements/base.txt" ]] && [[ -f "$search_dir/requirements/development.txt" ]]; then
                project_root="$search_dir"
                print_success "Found requirements files in: $project_root"
                requirements_found=true
                break
            fi
            search_dir="$(dirname "$search_dir")"
            if [[ "$search_dir" == "/" ]]; then
                break
            fi
        done
    fi
    
    # Method 4: Search from home directory
    if [[ "$requirements_found" == false ]]; then
        print_status "Searching for project root from home directory..."
        local home_dir="$HOME"
        
        # Look for common project locations
        local common_locations=(
            "$home_dir/nemobotter-python"
            "$home_dir/code/nemobotter-python"
            "$home_dir/Work/nemobotter-python"
            "$home_dir/projects/nemobotter-python"
        )
        
        for location in "${common_locations[@]}"; do
            if [[ -f "$location/requirements/base.txt" ]] && [[ -f "$location/requirements/development.txt" ]]; then
                project_root="$location"
                print_success "Found requirements files in: $project_root"
                requirements_found=true
                break
            fi
        done
    fi
    
    if [[ "$requirements_found" == false ]]; then
        print_error "Could not find project root with requirements files"
        print_error "Current working directory: $(pwd)"
        print_error "Script path: ${BASH_SOURCE[0]}"
        print_error "Please run this script from the project root directory"
        print_error "Or ensure the project structure is correct"
        exit 1
    fi
    
    # Store project root for use in other functions
    export PROJECT_ROOT="$project_root"
    print_status "Project root set to: $PROJECT_ROOT"
}

# Function to install system packages
install_system_packages() {
    print_status "Installing system packages..."
    
    # Update package list
    print_status "Updating package list..."
    sudo apt update
    
    # Install required system packages
    print_status "Installing required system packages..."
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        wget \
        curl \
        git
    
    print_success "System packages installed"
}

# Function to install external libraries using improved modular system
install_external_libraries() {
    print_status "Installing external libraries..."
    
    # Get the setup scripts directory
    local setup_dir="$PROJECT_ROOT/scripts/setup"
    
    if [[ ! -d "$setup_dir" ]]; then
        print_error "Setup scripts directory not found: $setup_dir"
        return 1
    fi
    
    # Run the master installation script
    print_status "Running external libraries installation..."
    if bash "$setup_dir/external_libs.sh" install; then
        print_success "External libraries installed successfully"
    else
        print_warning "Some external libraries may not have installed correctly"
        print_status "You can run individual library installations later:"
        print_status "  bash scripts/setup/cuda.sh install"
        print_status "  bash scripts/setup/cudnn.sh install"
        print_status "  bash scripts/setup/talib.sh install"
        print_status "  bash scripts/setup/lightgbm.sh install"
    fi
}

# Function to create virtual environment
create_virtual_env() {
    print_status "Creating virtual environment '.venv' in home directory..."
    
    # Get home directory
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    
    # Check if virtual environment already exists
    if [[ -d "$venv_path" ]]; then
        print_warning "Virtual environment '.venv' already exists"
        print_status "Using existing virtual environment"
    else
        # Create new virtual environment in home directory
        python3 -m venv "$venv_path"
        print_success "Virtual environment '.venv' created in $home_dir"
    fi
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Get home directory and venv path
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    
    # Use the project root from validation
    local project_root="$PROJECT_ROOT"
    
    print_status "Project root: $project_root"
    print_status "Virtual environment: $venv_path"
    
    # Upgrade pip, setuptools, wheel
    print_status "Upgrading pip, setuptools, wheel..."
    "$venv_path/bin/pip" install --upgrade pip setuptools wheel
    
    # Install base requirements
    print_status "Installing base requirements..."
    print_status "Requirements file: $project_root/requirements/base.txt"
    "$venv_path/bin/pip" install -r "$project_root/requirements/base.txt"
    
    # Install development requirements
    print_status "Installing development requirements..."
    print_status "Requirements file: $project_root/requirements/development.txt"
    "$venv_path/bin/pip" install -r "$project_root/requirements/development.txt"
    
    # Install GPU packages with NVIDIA PyPI index
    print_status "Installing GPU packages..."
    print_status "Requirements file: $project_root/requirements/gpu.txt"
    "$venv_path/bin/pip" install -r "$project_root/requirements/gpu.txt" --extra-index-url=https://pypi.nvidia.com
    
    # Fix NumPy compatibility issues
    fix_numpy_compatibility
    
    # Set up Python path for the project
    setup_python_path
    
    print_success "Python packages installed"
}

# Function to fix NumPy compatibility issues
fix_numpy_compatibility() {
    print_status "Checking and fixing NumPy compatibility issues..."
    
    # Get home directory and venv path
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    
    # Check current NumPy version
    local numpy_version=$("$venv_path/bin/python" -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    
    if [[ -n "$numpy_version" ]]; then
        print_status "Current NumPy version: $numpy_version"
        
        # Check if NumPy 2.x is installed (which causes compatibility issues)
        if [[ "$numpy_version" == 2.* ]]; then
            print_warning "NumPy 2.x detected - downgrading to 1.x for compatibility"
            
            # Downgrade NumPy to 1.x
            "$venv_path/bin/pip" install "numpy>=1.24.0,<2.0.0" --force-reinstall
            
            # Reinstall packages that might have NumPy 2.x compatibility issues
            print_status "Reinstalling packages with NumPy 1.x compatibility..."
            "$venv_path/bin/pip" install --force-reinstall \
                torch \
                tensorflow \
                scikit-learn \
                pandas \
                scipy
            
            print_success "NumPy compatibility issues fixed"
        else
            print_success "NumPy version is compatible (1.x)"
        fi
    else
        print_warning "Could not determine NumPy version"
    fi
}

# Function to set up Python path for the project
setup_python_path() {
    print_status "Setting up Python path for the project..."
    
    # Get home directory and venv path
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    local project_root="$PROJECT_ROOT"
    
    # Create a .pth file in the virtual environment to add the project root to Python path
    local site_packages_dir=$("$venv_path/bin/python" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    
    if [[ -n "$site_packages_dir" ]] && [[ -d "$site_packages_dir" ]]; then
        local pth_file="$site_packages_dir/nemobotter.pth"
        
        # Add project root to Python path (for relative imports to work)
        echo "$project_root" > "$pth_file"
        
        print_success "Python path configured: $project_root added to Python path"
        print_debug "Created .pth file: $pth_file"
    else
        print_warning "Could not determine site-packages directory"
    fi
    
    # Also add to PYTHONPATH environment variable for current session
    export PYTHONPATH="$project_root:$PYTHONPATH"
    
    # Create a startup script in the virtual environment to set PYTHONPATH
    local activate_script="$venv_path/bin/activate"
    local activate_script_content="$venv_path/bin/activate.nemobotter"
    
    # Create a custom activation script that sets PYTHONPATH
    cat > "$activate_script_content" << EOF
#!/bin/bash
# Nemobotter Python path setup
export PYTHONPATH="$project_root:\$PYTHONPATH"
echo "Nemobotter Python path configured: $project_root"
EOF
    
    chmod +x "$activate_script_content"
    
    # Add source command to the main activate script if not already present
    if ! grep -q "activate.nemobotter" "$activate_script"; then
        echo "" >> "$activate_script"
        echo "# Nemobotter Python path setup" >> "$activate_script"
        echo "source \"\$VIRTUAL_ENV/bin/activate.nemobotter\"" >> "$activate_script"
    fi
    
    print_success "Python path set up for current session and virtual environment activation"
    
    # Verify Python path is working
    verify_python_path
}

# Function to verify Python path is working correctly
verify_python_path() {
    print_status "Verifying Python path setup..."
    
    # Get home directory and venv path
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    local project_root="$PROJECT_ROOT"
    
    # Test if we can import from the project
    local test_script="/tmp/test_python_path.py"
    cat > "$test_script" << EOF
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

try:
    from src.utils.logger import get_logger
    print("✅ Successfully imported src.utils.logger")
except ImportError as e:
    print(f"❌ Failed to import src.utils.logger: {e}")
    sys.exit(1)

try:
    from src.research.mlflow_tracker import MLflowTracker
    print("✅ Successfully imported src.research.mlflow_tracker")
except ImportError as e:
    print(f"❌ Failed to import src.research.mlflow_tracker: {e}")
    sys.exit(1)

print("✅ Python path verification successful")
EOF
    
    # Run the test
    if PYTHONPATH="$project_root:$PYTHONPATH" "$venv_path/bin/python" "$test_script"; then
        print_success "Python path verification passed"
    else
        print_warning "Python path verification failed - imports may not work correctly"
    fi
    
    # Clean up
    rm -f "$test_script"
}

# Function to test external libraries
test_external_libraries() {
    print_status "Testing external libraries..."
    
    # Get the setup scripts directory
    local setup_dir="$PROJECT_ROOT/scripts/setup"
    
    if [[ ! -d "$setup_dir" ]]; then
        print_error "Setup scripts directory not found: $setup_dir"
        return 1
    fi
    
    # Run the external libraries test
    print_status "Running external libraries tests..."
    if bash "$setup_dir/external_libs.sh" test; then
        print_success "External libraries tests passed"
    else
        print_warning "Some external libraries tests failed"
        print_status "You can run individual library tests:"
        print_status "  bash scripts/setup/cuda.sh test"
        print_status "  bash scripts/setup/cudnn.sh test"
        print_status "  bash scripts/setup/talib.sh test"
        print_status "  bash scripts/setup/lightgbm.sh test"
    fi
}

# Function to test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Get home directory and venv path
    local home_dir="$HOME"
    local venv_path="$home_dir/.venv"
    
    # Use the project root from validation
    local project_root="$PROJECT_ROOT"
    
    # Test research tools
    print_status "Running research tools test..."
    PYTHONPATH="$project_root:$PYTHONPATH" "$venv_path/bin/python" "$project_root/scripts/test_research_tools.py"
    
    print_success "Installation test completed"
}

# Function to show usage information
show_usage() {
    print_success "Setup completed successfully!"
    echo
    echo -e "${GREEN}Directory Structure:${NC}"
    echo -e "  WSL Home (~): Virtual environment and external libraries"
    echo -e "  Project Dir: Research tools and project-specific code"
    echo
    echo -e "${GREEN}Virtual environment '.venv' is ready in your home directory!${NC}"
    echo
    echo -e "${CYAN}To activate the virtual environment:${NC}"
    echo -e "  source ~/.venv/bin/activate"
    echo
    echo -e "${CYAN}To check external libraries status:${NC}"
    echo -e "  bash scripts/setup/external_libs.sh status"
    echo
    echo -e "${CYAN}To test external libraries:${NC}"
    echo -e "  bash scripts/setup/external_libs.sh test"
    echo
    echo -e "${CYAN}To run the research tools test:${NC}"
    echo -e "  ~/.venv/bin/python scripts/test_research_tools.py"
    echo
    echo -e "${CYAN}To run the trading bot:${NC}"
    echo -e "  ~/.venv/bin/python -m src.web_interface.app"
    echo
    echo -e "${CYAN}Important:${NC}"
    echo -e "  - Run this script from the project root directory (where requirements/ folder is located)"
    echo -e "  - Run test commands from the project root directory"
    echo
    echo -e "${CYAN}External Libraries:${NC}"
    echo -e "  - External libraries are installed using modular scripts in scripts/setup/"
    echo -e "  - Check status: bash scripts/setup/external_libs.sh status"
    echo -e "  - Run tests: bash scripts/setup/external_libs.sh test"
    echo -e "  - Individual libraries: scripts/setup/[cuda|cudnn|talib|lightgbm].sh"
    echo -e "  - Installation files: ~/.nemobotter-external/ and ~/.nemobotter-cuda-tools/"
    echo
    echo -e "${CYAN}To deactivate the virtual environment:${NC}"
    echo -e "  deactivate"
}

# Function to parse command line arguments
parse_args() {
    DEBUG_FLAG="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug)
                DEBUG_FLAG="true"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--debug] [--help]"
                echo "  --debug  - Enable detailed logging"
                echo "  --help   - Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}NemoBotter Trading System - Setup Script${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo
    
    # Parse command line arguments
    parse_args "$@"
    
    # Check if not running as root
    check_root
    
    # Check system requirements
    check_system
    
    # Install system packages
    install_system_packages
    
    # Install external libraries using improved modular system
    install_external_libraries
    
    # Create virtual environment
    create_virtual_env
    
    # Install Python packages
    install_python_packages
    
    # Test external libraries
    test_external_libraries
    
    # Test installation
    test_installation
    
    # Show usage information
    show_usage
}

# Run main function
main "$@" 