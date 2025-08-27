#!/bin/bash
# T-Bot Trading System - Setup Validation Script

echo "======================================================"
echo "T-Bot Trading System - Setup Validation"
echo "======================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track validation status
ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2 - File missing: $1"
        ((ERRORS++))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2 - Directory missing: $1"
        ((ERRORS++))
    fi
}

# Function to check environment variable
check_env() {
    if [ -n "${!1}" ]; then
        echo -e "${GREEN}✓${NC} Environment variable $1 is set"
    else
        echo -e "${YELLOW}⚠${NC} Environment variable $1 is not set"
        ((WARNINGS++))
    fi
}

echo ""
echo "1. Checking Core Files..."
echo "--------------------------"
check_file "docker/docker-compose.yml" "Docker Compose configuration"
check_file "docker/Dockerfile.backend" "Backend Dockerfile"
check_file "docker/Dockerfile.gpu" "GPU Dockerfile"
check_file "requirements.txt" "Python requirements"
check_file ".env" "Environment configuration"
check_file "alembic.ini" "Database migration config"
check_file "Makefile" "Makefile configuration"

echo ""
echo "2. Checking Source Code Structure..."
echo "-------------------------------------"
check_dir "src/core" "Core module"
check_dir "src/utils" "Utils module"
check_dir "src/error_handling" "Error handling module"
check_dir "src/database" "Database module"
check_dir "src/exchanges" "Exchanges module"
check_dir "src/risk_management" "Risk management module"
check_dir "src/strategies" "Strategies module"
check_dir "src/web_interface" "Web interface module"

echo ""
echo "3. Checking Additional Modules..."
echo "----------------------------------"
check_dir "src/ml" "Machine Learning module"
check_dir "src/data" "Data processing module"
check_dir "src/execution" "Execution module"
check_dir "src/optimization" "Optimization module"
check_dir "src/backtesting" "Backtesting module"
check_dir "src/capital_management" "Capital management module"
check_dir "src/bot_management" "Bot management module"
check_file "src/utils/gpu_utils.py" "GPU utilities"

echo ""
echo "4. Checking Docker Configuration..."
echo "------------------------------------"
check_dir "docker/configs" "Docker configs"
check_dir "docker/services" "Docker services"
check_file "docker/services/postgresql/init.sql" "PostgreSQL init script"

echo ""
echo "5. Checking Test Structure..."
echo "------------------------------"
check_dir "tests/unit" "Unit tests"
check_dir "tests/integration" "Integration tests"
check_file "pytest.ini" "Pytest configuration"

echo ""
echo "6. Checking Documentation..."
echo "-----------------------------"
check_file "README.md" "Main README"
check_file "docs/SPECIFICATIONS.md" "Project specifications"
check_file "docs/CODING_STANDARDS.md" "Coding standards"
check_file "CLAUDE.md" "Claude configuration"

echo ""
echo "7. Checking Python Environment..."
echo "----------------------------------"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python installed: $PYTHON_VERSION"
    
    # Check if version is 3.10+
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓${NC} Python version meets requirements (3.10+)"
    else
        echo -e "${YELLOW}⚠${NC} Python version should be 3.10 or higher"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} Python not found"
    ((ERRORS++))
fi

echo ""
echo "8. Checking Docker Installation..."
echo "-----------------------------------"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo -e "${GREEN}✓${NC} Docker installed: $DOCKER_VERSION"
else
    echo -e "${RED}✗${NC} Docker not installed"
    ((ERRORS++))
fi

if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose available"
else
    echo -e "${RED}✗${NC} Docker Compose not available"
    ((ERRORS++))
fi

echo ""
echo "======================================================"
echo "Validation Summary"
echo "======================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! System is ready for deployment.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Validation completed with $WARNINGS warning(s).${NC}"
    echo "The system can run but review warnings for optimal setup."
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo "Please fix the errors before proceeding."
    exit 1
fi