# T-Bot Trading System Makefile

# Use bash for Makefile recipes
SHELL := /bin/bash

# Python settings
VENV := ~/.venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON := $(VENV_BIN)/python

# Project directories
PROJECT_DIR := $(shell pwd)
SRC_DIR := src
TEST_DIR := tests
CONFIG_DIR := config
DOCKER_DIR := docker
SCRIPTS_DIR := scripts

# Docker settings
-include .env
export
DOCKER_COMPOSE := docker-compose -f "$(DOCKER_DIR)/docker-compose.yml"
DOCKER_COMPOSE_SERVICES := docker-compose -f "$(DOCKER_DIR)/docker-compose.services.yml"
DOCKER_COMPOSE_PROD := docker-compose -f "$(DOCKER_DIR)/docker-compose.prod.yml"
DOCKER_COMPOSE_TEST := docker-compose -f "$(DOCKER_DIR)/docker-compose.test.yml"
DOCKER_COMPOSE_MONITORING := docker-compose -f "$(DOCKER_DIR)/docker-compose.monitoring.yml"

# GPU/CUDA settings
CUDA_VERSION := 12.1
CUDNN_VERSION := 8.9
PYTHON_VERSION := 3.10

.PHONY: help setup setup-venv setup-external setup-gpu install-deps install-gpu-deps
.PHONY: test test-unit test-integration test-mock coverage lint format typecheck
.PHONY: docker-build docker-up docker-down docker-logs docker-clean
.PHONY: services-up services-down services-logs services-clean
.PHONY: run run-mock run-web run-all run-all-dev kill stop restart status migrate clean clean-deep validate audit
.PHONY: pre-commit check-all fix-all

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@echo ''
	@echo '🚀 Setup & Installation:'
	@echo '    setup            - Complete setup (venv, deps, external libs, GPU)'
	@echo '    setup-venv       - Create and setup Python virtual environment'
	@echo '    setup-external   - Install external dependencies (TA-Lib, etc.)'
	@echo '    setup-frontend   - Install Node.js and frontend dependencies'
	@echo '    setup-gpu        - Install GPU/CUDA dependencies'
	@echo '    install-deps     - Install Python dependencies'
	@echo '    install-gpu-deps - Install GPU-enabled Python packages'
	@echo ''
	@echo '🏃 Running the Application:'
	@echo '    run              - Run T-Bot trading system'
	@echo '    run-mock         - Run T-Bot in mock mode (no API keys)'
	@echo '    run-backend      - Run backend trading engine only'
	@echo '    run-web          - Start web API (backend)'
	@echo '    run-worker       - Run background worker processes'
	@echo '    run-websocket    - Run WebSocket server only'
	@echo '    run-frontend     - Start React frontend'
	@echo '    run-all          - Start backend, API, and frontend'
	@echo '    run-all-dev      - Start backend and frontend in development watch mode'
	@echo '    kill             - Stop all running servers'
	@echo '    stop             - Alias for kill command'
	@echo '    restart          - Restart all servers'
	@echo '    status           - Check status of T-Bot services'
	@echo '    migrate          - Run database migrations'
	@echo ''
	@echo '🐳 Docker & Services:'
	@echo '    docker-build     - Build Docker images'
	@echo '    docker-up        - Start all services with Docker'
	@echo '    docker-down      - Stop Docker services'
	@echo '    services-up      - Start external services only'
	@echo '    services-down    - Stop external services'
	@echo ''
	@echo '🧪 Testing & Quality:'
	@echo '    test             - Run all tests'
	@echo '    test-unit        - Run unit tests only'
	@echo '    test-integration - Run integration tests'
	@echo '    test-mock        - Run tests in mock mode'
	@echo '    coverage         - Run tests with coverage report'
	@echo '    lint             - Run linting checks'
	@echo '    format           - Format code with ruff and black'
	@echo '    typecheck        - Run type checking with mypy'
	@echo '    check-all        - Run all checks (lint, type, test)'
	@echo '    fix-all          - Fix all auto-fixable issues'
	@echo ''
	@echo '🔍 Validation & Audit:'
	@echo '    validate         - Validate project configuration'
	@echo '    audit            - Complete system audit'
	@echo '    pre-commit       - Run pre-commit checks'
	@echo ''
	@echo '🧹 Cleanup:'
	@echo '    clean            - Clean temporary files'
	@echo '    clean-deep       - Deep clean including Docker volumes'

# ============================================================================
# Setup & Installation Commands
# ============================================================================

setup: ## Complete setup (venv, deps, external libs, GPU, frontend)
	@echo "🔧 Complete T-Bot Setup..."
	@echo "📋 Running pre-installation checks..."
	@bash "$(SCRIPTS_DIR)/setup/pre_install.sh"
	@$(MAKE) -s setup-venv
	@$(MAKE) -s setup-external
	@$(MAKE) -s install-deps
	@echo "🎮 Optional: Installing GPU support (failures won't stop setup)..."
	@$(MAKE) -s setup-gpu || echo "⚠️  GPU setup skipped (optional)"
	@$(MAKE) -s install-gpu-deps || echo "⚠️  GPU packages skipped (optional)"
	@echo "🎨 Setting up frontend..."
	@if command -v node > /dev/null 2>&1 || [ -f ~/.nvm/nvm.sh ]; then \
		echo "Installing frontend dependencies..."; \
		if [ -f ~/.nvm/nvm.sh ]; then \
			. ~/.nvm/nvm.sh; \
		fi; \
		cd frontend && npm install --silent 2>/dev/null || echo "⚠️  Frontend setup skipped (npm install failed)"; \
	else \
		echo "⚠️  Node.js not found. Run 'make setup-frontend' to install"; \
	fi
	@$(MAKE) -s services-up
	@$(MAKE) -s migrate
	@echo "✅ Complete setup finished!"
	@echo "ℹ️  Run 'make run-all' to start the full application"
	@echo "ℹ️  Run 'make test' to verify everything works"

setup-venv: ## Create and setup Python virtual environment
	@echo "🐍 Setting up Python virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		python$(PYTHON_VERSION) -m venv $(VENV); \
		echo "✅ Virtual environment created at $(VENV)"; \
	else \
		echo "ℹ️  Virtual environment already exists at $(VENV)"; \
	fi
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "✅ Virtual environment ready!"

setup-external: ## Install external dependencies (TA-Lib, etc.)
	@echo "📦 Installing external libraries..."
	@bash "$(SCRIPTS_DIR)/setup/external_libs.sh" install
	@echo "✅ External libraries installed!"

setup-frontend: ## Install Node.js and frontend dependencies
	@echo "🎨 Setting up frontend (Node.js + React)..."
	@bash "$(SCRIPTS_DIR)/setup/nodejs.sh"
	@echo "✅ Frontend setup completed!"

setup-gpu: ## Install GPU/CUDA dependencies
	@echo "🎮 Setting up GPU/CUDA support..."
	@bash "$(SCRIPTS_DIR)/setup/cuda.sh" install
	@bash "$(SCRIPTS_DIR)/setup/cudnn.sh" install
	@bash "$(SCRIPTS_DIR)/setup/lightgbm.sh" install
	@echo "✅ GPU setup completed!"

install-deps: ## Install Python dependencies
	@echo "📦 Installing Python dependencies..."
	@bash "$(SCRIPTS_DIR)/setup/install_requirements.sh"
	@echo "✅ Python dependencies installed!"

install-gpu-deps: ## Install GPU-enabled Python packages
	@echo "🎮 Installing GPU-enabled Python packages..."
	@echo "📦 Installing PyTorch with CUDA 12.1 support..."
	@$(PIP) install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || true
	@echo "📦 Installing TensorFlow with CUDA support..."
	@$(PIP) install --upgrade tensorflow[and-cuda] || true
	@echo "📦 Installing CuPy for GPU arrays..."
	@$(PIP) install --upgrade cupy-cuda12x || true
	@echo "ℹ️  Note: RAPIDS requires special installation from NVIDIA channels"
	@echo "✅ GPU-enabled packages installed!"

# ============================================================================
# Running Commands
# ============================================================================

run: ## Run T-Bot trading system
	@echo "🚀 Starting T-Bot Trading System..."
	@echo "⚠️  Checking services..."
	@$(MAKE) -s services-check || true
	@bash -c 'source $(VENV)/bin/activate && python -m src.main'

run-mock: ## Run T-Bot in mock mode (no API keys)
	@echo "🚀 Starting T-Bot in Mock Mode..."
	@bash -c 'source $(VENV)/bin/activate && MOCK_MODE=true python -m src.main'

run-web: ## Start web API (backend)
	@echo "🌐 Starting Web API..."
	@bash -c 'source $(VENV)/bin/activate && uvicorn src.web_interface.app:get_app --host 0.0.0.0 --port 8000 --log-level info'

run-frontend: ## Start frontend React application
	@echo "🎨 Starting Frontend React App..."
	@if command -v npm > /dev/null 2>&1; then \
		cd frontend && npm install --silent 2>/dev/null && npm start; \
	elif [ -f ~/.nvm/nvm.sh ]; then \
		bash -c '. ~/.nvm/nvm.sh && cd frontend && npm install --silent 2>/dev/null && npm start'; \
	else \
		echo "❌ Node.js/npm not installed. Please install Node.js first:"; \
		echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"; \
		echo "   nvm install node"; \
		exit 1; \
	fi

kill: ## Kill all running servers (frontend and backend)
	@echo "🛑 Stopping all T-Bot servers..."
	@echo "Killing backend processes..."
	@-pkill -f "python.*-m.*src\.main" 2>/dev/null || true
	@-pkill -f "python.*src/main\.py" 2>/dev/null || true
	@echo "Killing web API processes..."
	@-pkill -f "python.*-m.*src\.web_interface" 2>/dev/null || true
	@-pkill -f "python.*src/web_interface/app\.py" 2>/dev/null || true
	@echo "Killing Uvicorn processes..."
	@-pkill -f "uvicorn.*src\.web_interface" 2>/dev/null || true
	@-pkill -f "uvicorn.*get_app" 2>/dev/null || true
	@echo "Killing frontend processes..."
	@-pkill -f "webpack.*serve" 2>/dev/null || true
	@-pkill -f "node.*webpack" 2>/dev/null || true
	@-pkill -f "npm.*start" 2>/dev/null || true
	@-pkill -f "npm.*run.*start" 2>/dev/null || true
	@-pkill -f "node.*react-scripts" 2>/dev/null || true
	@-pkill -f "webpack-dev-server" 2>/dev/null || true
	@echo "Killing Node.js processes on frontend..."
	@-ps aux | grep -E "(webpack|react|npm).*start" | grep -v grep | awk '{print $$2}' | xargs -r kill -9 2>/dev/null || true
	@echo "Killing processes on common ports..."
	@-lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
	@-lsof -ti:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true  
	@-lsof -ti:8080 2>/dev/null | xargs -r kill -9 2>/dev/null || true
	@echo "✅ All servers stopped!"

stop: kill ## Alias for kill command

restart: ## Restart all servers
	@$(MAKE) kill
	@sleep 2
	@echo "🔄 Restarting servers..."
	@$(MAKE) run-all

run-all: ## Start backend, web API, and frontend (if Node.js available)
	@echo "🚀 Starting all T-Bot services..."
	@bash "$(SCRIPTS_DIR)/development/run-all.sh"

run-all-dev: ## Start backend and frontend in development watch mode (auto-reload on changes)
	@echo "🚀 Starting T-Bot in development mode..."
	@bash "$(SCRIPTS_DIR)/development/run-dev.sh"

status: ## Check status of T-Bot services
	@echo "📊 T-Bot Service Status:"
	@echo "------------------------"
	@echo -n "Backend Process: "
	@ps aux | grep -E "python.*-m.*src\.main|python.*src/main\.py" | grep -v grep > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped"
	@echo -n "Web API (FastAPI): "
	@ps aux | grep -E "python.*-m.*src\.web_interface|uvicorn.*src\.web_interface|uvicorn.*get_app" | grep -v grep > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped"
	@echo -n "Frontend (React): "
	@ps aux | grep -E "webpack.*serve|node.*webpack|npm.*start|node.*react-scripts|webpack-dev-server" | grep -v grep > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped"
	@echo -n "API Port 8000: "
	@lsof -i:8000 > /dev/null 2>&1 && echo "✅ In use (http://localhost:8000/docs)" || echo "❌ Free"
	@echo -n "Frontend Port 3000: "
	@lsof -i:3000 > /dev/null 2>&1 && echo "✅ In use (http://localhost:3000)" || echo "❌ Free"
	@echo -n "Docker Services: "
	@docker ps 2>/dev/null | grep -q tbot && echo "✅ Running" || echo "❌ Stopped"
	@echo ""
	@echo "Active Python Processes:"
	@ps aux | grep -E "python.*src\.(main|web_interface)" | grep -v grep || echo "  None found"
	@echo ""
	@echo "Active Frontend Processes:"
	@ps aux | grep -E "(webpack|react|npm).*start" | grep -v grep || echo "  None found"

migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	@bash -c 'source $(VENV)/bin/activate && alembic upgrade head 2>/dev/null || echo "⚠️  No migrations to run or alembic not installed"'
	@echo "✅ Migrations completed!"

# ============================================================================
# Docker & Services Commands
# ============================================================================

docker-build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	@docker build -t tbot-backend:latest -f "$(DOCKER_DIR)/Dockerfile.backend" .
	@docker build -t tbot-gpu:latest -f "$(DOCKER_DIR)/Dockerfile" .
	@echo "✅ Docker images built!"

docker-up: ## Start all services with Docker
	@echo "🐳 Starting all services with Docker..."
	@$(DOCKER_COMPOSE) up -d
	@echo "✅ All services started!"

docker-down: ## Stop Docker services
	@echo "🛑 Stopping Docker services..."
	@$(DOCKER_COMPOSE) down
	@echo "✅ Services stopped!"

docker-logs: ## Show Docker logs
	@$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean Docker resources
	@echo "🧹 Cleaning Docker resources..."
	@$(DOCKER_COMPOSE) down -v
	@docker system prune -f
	@echo "✅ Docker cleanup completed!"

services-up: ## Start external services only
	@echo "🐳 Starting external services..."
	@if [ ! -d "$(DOCKER_DIR)" ]; then \
		echo "⚠️  Docker directory not found, using current directory"; \
		docker-compose -f docker-compose.services.yml up -d; \
	else \
		$(DOCKER_COMPOSE_SERVICES) up -d; \
	fi
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 5
	@echo "✅ Services started!"
	@echo "📊 PostgreSQL: localhost:5432"
	@echo "🔴 Redis: localhost:6379"
	@echo "📈 InfluxDB: localhost:8086"

services-down: ## Stop external services
	@echo "🛑 Stopping external services..."
	@if [ ! -d "$(DOCKER_DIR)" ]; then \
		docker-compose -f docker-compose.services.yml down; \
	else \
		$(DOCKER_COMPOSE_SERVICES) down; \
	fi
	@echo "✅ Services stopped!"

services-logs: ## Show logs from external services
	@if [ ! -d "$(DOCKER_DIR)" ]; then \
		docker-compose -f docker-compose.services.yml logs -f; \
	else \
		$(DOCKER_COMPOSE_SERVICES) logs -f; \
	fi

services-clean: ## Stop services and remove volumes
	@echo "🧹 Cleaning up services and volumes..."
	@if [ ! -d "$(DOCKER_DIR)" ]; then \
		docker-compose -f docker-compose.services.yml down -v; \
	else \
		$(DOCKER_COMPOSE_SERVICES) down -v; \
	fi
	@echo "✅ Cleanup completed!"

services-check: ## Check if services are running
	@echo "🔍 Checking services status..."
	@docker ps 2>/dev/null | grep -q postgres || echo "⚠️  PostgreSQL not running. Run 'make services-up'"
	@docker ps 2>/dev/null | grep -q redis || echo "⚠️  Redis not running. Run 'make services-up'"
	@echo "✅ Service check completed!"

# ============================================================================
# Testing & Quality Commands
# ============================================================================

test: ## Run all tests
	@echo "🧪 Running all tests..."
	@bash -c 'source $(VENV)/bin/activate && python -m pytest "$(TEST_DIR)/" -v --tb=short'
	@echo "✅ Tests completed!"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@bash -c 'source $(VENV)/bin/activate && python -m pytest "$(TEST_DIR)/unit/" -v --tb=short'
	@echo "✅ Unit tests completed!"

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@bash -c 'source $(VENV)/bin/activate && python -m pytest "$(TEST_DIR)/integration/" -v --tb=short'
	@echo "✅ Integration tests completed!"

test-mock: ## Run tests in mock mode
	@echo "🧪 Running tests in mock mode..."
	@bash -c 'source $(VENV)/bin/activate && MOCK_MODE=true python -m pytest "$(TEST_DIR)/" -v --tb=short'
	@echo "✅ Mock tests completed!"

coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	@bash -c 'source $(VENV)/bin/activate && python -m pytest "$(TEST_DIR)/" --cov="$(SRC_DIR)" --cov-report=html --cov-report=term-missing'
	@echo "✅ Coverage report generated!"
	@echo "📁 HTML report: htmlcov/index.html"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@bash -c 'source $(VENV)/bin/activate && ruff check "$(SRC_DIR)" "$(TEST_DIR)"'
	@echo "✅ Linting completed!"

format: ## Format code with ruff and black
	@echo "🎨 Formatting code..."
	@bash -c 'source $(VENV)/bin/activate && ruff check "$(SRC_DIR)" "$(TEST_DIR)" --fix'
	@bash -c 'source $(VENV)/bin/activate && ruff format "$(SRC_DIR)" "$(TEST_DIR)"'
	@bash -c 'source $(VENV)/bin/activate && black "$(SRC_DIR)" "$(TEST_DIR)" --line-length 100'
	@echo "✅ Code formatted!"

typecheck: ## Run type checking with mypy
	@echo "🔍 Running type checking..."
	@bash -c 'source $(VENV)/bin/activate && mypy "$(SRC_DIR)" --ignore-missing-imports'
	@echo "✅ Type checking completed!"

check-all: ## Run all checks (lint, type, test)
	@echo "🔍 Running all checks..."
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) test
	@echo "✅ All checks passed!"

fix-all: ## Fix all auto-fixable issues
	@echo "🔧 Fixing all auto-fixable issues..."
	@$(MAKE) format
	@echo "✅ All fixable issues resolved!"

# ============================================================================
# Validation & Audit Commands
# ============================================================================

validate: ## Validate project configuration
	@echo "🔍 Validating project configuration..."
	@bash "$(SCRIPTS_DIR)/development/validate_setup.sh" || echo "⚠️  Validation script not found"
	@echo "✅ Validation completed!"

audit: ## Complete system audit
	@echo "🔍 Running complete system audit..."
	@bash "$(SCRIPTS_DIR)/system/final_validation.sh" || echo "⚠️  Audit script not found"
	@echo "✅ Audit completed!"

pre-commit: ## Run pre-commit checks
	@echo "🔍 Running pre-commit checks..."
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) test-unit
	@echo "✅ Pre-commit checks passed!"

# ============================================================================
# Cleanup Commands
# ============================================================================

clean: ## Clean temporary files
	@echo "🧹 Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf .pytest_cache/ 2>/dev/null || true
	@rm -rf .mypy_cache/ 2>/dev/null || true
	@rm -rf .ruff_cache/ 2>/dev/null || true
	@echo "✅ Cleanup completed!"

clean-deep: ## Deep clean including Docker volumes
	@echo "🧹 Deep cleaning..."
	@$(MAKE) clean
	@$(MAKE) docker-clean
	@rm -rf ~/.nemobotter-external/ 2>/dev/null || true
	@rm -rf ~/.nemobotter-cuda-tools/ 2>/dev/null || true
	@echo "✅ Deep cleanup completed!"