#!/bin/bash
#
# Node.js installation script for T-Bot frontend
#
# This script installs Node.js and npm for running the React frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}T-Bot Frontend Setup - Node.js Installation${NC}"
echo "================================================"

# Check if Node.js is already installed
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${YELLOW}Node.js is already installed: ${NODE_VERSION}${NC}"
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        echo -e "${YELLOW}npm is already installed: ${NPM_VERSION}${NC}"
        exit 0
    fi
fi

echo -e "${GREEN}Installing Node.js and npm...${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Detected Debian/Ubuntu system"
        
        # Install Node.js 18.x (LTS)
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
    elif command -v yum &> /dev/null; then
        # RedHat/CentOS
        echo "Detected RedHat/CentOS system"
        curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
        sudo yum install -y nodejs
        
    else
        echo -e "${RED}Unsupported Linux distribution${NC}"
        echo "Please install Node.js manually from: https://nodejs.org/"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS"
    
    if command -v brew &> /dev/null; then
        brew install node
    else
        echo -e "${YELLOW}Homebrew not found. Installing Homebrew first...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install node
    fi
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo -e "${YELLOW}Windows detected. Please install Node.js from:${NC}"
    echo "https://nodejs.org/en/download/"
    exit 1
    
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

# Verify installation
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ Node.js installed successfully: ${NODE_VERSION}${NC}"
    echo -e "${GREEN}✅ npm installed successfully: ${NPM_VERSION}${NC}"
    
    # Install frontend dependencies
    echo -e "${GREEN}Installing frontend dependencies...${NC}"
    cd "$(dirname "$0")/../../frontend"
    npm install
    
    echo -e "${GREEN}✅ Frontend setup complete!${NC}"
    echo -e "${YELLOW}You can now run 'make run-frontend' or 'make run-all'${NC}"
else
    echo -e "${RED}❌ Installation failed. Please install Node.js manually.${NC}"
    exit 1
fi