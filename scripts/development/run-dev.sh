#!/bin/bash
# Development mode runner for T-Bot with auto-reload
# This script runs backend and frontend with automatic reloading on file changes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$HOME/.venv"
NODE_PATH="$HOME/.nvm/versions/node/v18.19.0/bin"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}⚠️  Stopping all services...${NC}"
    
    # Kill all child processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Kill specific ports if still in use
    lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM EXIT

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Run 'make setup' to create it${NC}"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Add Node.js to PATH if available
if [ -d "$NODE_PATH" ]; then
    export PATH="$NODE_PATH:$PATH"
fi

echo -e "${BLUE}🚀 Starting T-Bot in development mode with auto-reload...${NC}"
echo -e "${YELLOW}📝 Changes to source files will automatically reload${NC}\n"

# Kill any existing processes on our ports
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2

# Start FastAPI with auto-reload
echo -e "${GREEN}1️⃣ Starting Web API (FastAPI with auto-reload)...${NC}"
cd "$PROJECT_DIR"
uvicorn src.web_interface.app:get_app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src \
    --log-level info 2>&1 | sed 's/^/[API] /' &
API_PID=$!

# Wait for API to start
sleep 3

# Start Backend in mock mode (we'll use a simple loop to restart on changes)
echo -e "${GREEN}2️⃣ Starting Backend (Python in mock mode)...${NC}"
cd "$PROJECT_DIR"

# Create a simple file watcher for backend
(
    while true; do
        echo -e "${BLUE}[BACKEND] Starting backend process...${NC}"
        MOCK_MODE=true python -m src.main 2>&1 | sed 's/^/[BACKEND] /' &
        BACKEND_PID=$!
        
        # Use inotifywait if available, otherwise use a simple loop
        if command -v inotifywait >/dev/null 2>&1; then
            # Watch for Python file changes
            inotifywait -r -e modify,create,delete --include '\.py$' src/ 2>/dev/null
        else
            # Fallback: check for file changes every 2 seconds
            LAST_MODIFIED=$(find src -name "*.py" -type f -exec stat -c %Y {} \; | sort -n | tail -1)
            while true; do
                sleep 2
                CURRENT_MODIFIED=$(find src -name "*.py" -type f -exec stat -c %Y {} \; | sort -n | tail -1)
                if [ "$LAST_MODIFIED" != "$CURRENT_MODIFIED" ]; then
                    LAST_MODIFIED=$CURRENT_MODIFIED
                    break
                fi
            done
        fi
        
        echo -e "${YELLOW}[BACKEND] Changes detected, restarting...${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    done
) &
BACKEND_WATCHER_PID=$!

# Wait a bit for backend to start
sleep 3

# Start Frontend if Node.js is available
if command -v npm >/dev/null 2>&1; then
    echo -e "${GREEN}3️⃣ Starting Frontend (React with hot-reload)...${NC}"
    cd "$PROJECT_DIR/frontend"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
    fi
    
    # Start the frontend (npm start already has hot-reload)
    PORT=3000 npm start 2>&1 | sed 's/^/[FRONTEND] /' &
    FRONTEND_PID=$!
else
    echo -e "${YELLOW}⚠️  Node.js not found. Frontend will not be started.${NC}"
    echo -e "${YELLOW}   Install with: make setup-frontend${NC}"
fi

echo -e "\n${GREEN}✅ All services started in development mode!${NC}\n"
echo -e "${BLUE}📌 Access points:${NC}"
echo -e "   • Frontend:    http://localhost:3000"
echo -e "   • API Docs:    http://localhost:8000/docs"
echo -e "   • API Health:  http://localhost:8000/health"
echo -e "   • WebSocket:   ws://localhost:8000/ws\n"
echo -e "${YELLOW}🔄 Auto-reload enabled:${NC}"
echo -e "   • FastAPI reloads on Python changes in src/"
echo -e "   • Backend restarts on Python changes in src/"
echo -e "   • React hot-reloads on frontend changes\n"
echo -e "${RED}🛑 Press Ctrl+C to stop all services${NC}\n"

# Wait for all background processes
wait