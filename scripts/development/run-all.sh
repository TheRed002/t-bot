#!/bin/bash
#
# Run all T-Bot services and handle graceful shutdown
#
# This script starts all services in the background and waits for them,
# properly handling SIGINT (Ctrl+C) to cleanly shut down all processes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Store PIDs of background processes
PIDS=()

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Virtual environment path
VENV="$HOME/.venv"

# Function to kill all background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down T-Bot services...${NC}"
    
    # Kill all stored PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping process $pid...${NC}"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Give processes time to shutdown gracefully
    sleep 2
    
    # Force kill if still running
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}Force stopping process $pid...${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # Also kill any remaining processes by name
    pkill -f "python.*-m.*src\.main" 2>/dev/null || true
    pkill -f "python.*src\.web_interface" 2>/dev/null || true
    pkill -f "webpack.*serve" 2>/dev/null || true
    pkill -f "node.*webpack" 2>/dev/null || true
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "$VENV" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV${NC}"
    echo -e "${YELLOW}Please run 'make setup' first${NC}"
    exit 1
fi

# Start web API only (includes backend functionality)
echo -e "${BLUE}Starting web API...${NC}"
source "$VENV/bin/activate"
MOCK_MODE=true python -c "import uvicorn; from src.web_interface.app import get_app; uvicorn.run(get_app(), host='0.0.0.0', port=8000, log_level='info')" 2>&1 | sed "s/^/\x1b[36m[API]\x1b[0m /" &
API_PID=$!
PIDS+=($API_PID)
echo -e "${GREEN}✅ Web API started (PID: $API_PID)${NC}"

sleep 2

# Start frontend if Node.js is available
if command -v npm > /dev/null 2>&1; then
    echo -e "${BLUE}Starting frontend...${NC}"
    cd frontend
    npm start 2>&1 | sed "s/^/\x1b[35m[FRONTEND]\x1b[0m /" &
    FRONTEND_PID=$!
    PIDS+=($FRONTEND_PID)
    cd ..
    echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
elif [ -f ~/.nvm/nvm.sh ]; then
    echo -e "${BLUE}Starting frontend (using nvm)...${NC}"
    bash -c "source ~/.nvm/nvm.sh && cd frontend && npm start 2>&1 | sed 's/^/\x1b[35m[FRONTEND]\x1b[0m /'" &
    FRONTEND_PID=$!
    PIDS+=($FRONTEND_PID)
    echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend not started (Node.js/npm not installed)${NC}"
    echo -e "${YELLOW}   To install Node.js:${NC}"
    echo -e "${YELLOW}   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash${NC}"
    echo -e "${YELLOW}   nvm install node${NC}"
fi

# Wait a moment for services to initialize
sleep 3

# Display service information
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ All T-Bot services are running!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📊 Backend:${NC} Integrated with Web API in mock mode"
echo -e "${BLUE}🌐 Web API:${NC} http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC} http://localhost:8000/docs"
if [ ! -z "$FRONTEND_PID" ]; then
    echo -e "${BLUE}🎨 Frontend:${NC} http://localhost:3000"
fi
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""

# Wait for all background processes
# This will keep the script running until interrupted
while true; do
    # Check if any of our processes are still running
    RUNNING=false
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING=true
            break
        fi
    done
    
    if [ "$RUNNING" = false ]; then
        echo -e "${YELLOW}All services have stopped${NC}"
        break
    fi
    
    # Sleep for a bit before checking again
    sleep 1
done

# Clean exit
cleanup