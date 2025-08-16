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
DOCKER_COMPOSE := docker-compose -f $(DOCKER_DIR)/docker-compose.yml
DOCKER_COMPOSE_SERVICES := docker-compose -f $(DOCKER_DIR)/docker-compose.services.yml
DOCKER_COMPOSE_FULL := docker-compose -f $(DOCKER_DIR)/docker-compose.full.yml

# GPU/CUDA settings
CUDA_VERSION := 12.1
CUDNN_VERSION := 8.9
PYTHON_VERSION := 3.10

.PHONY: help setup setup-venv setup-external setup-gpu install-deps install-gpu-deps
.PHONY: test test-unit test-integration test-mock coverage lint format typecheck
.PHONY: docker-build docker-up docker-down docker-logs docker-clean
.PHONY: services-up services-down services-logs services-clean
.PHONY: run run-mock run-web migrate clean clean-deep validate audit
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
	@echo '    setup-gpu        - Install GPU/CUDA dependencies'
	@echo '    install-deps     - Install Python dependencies'
	@echo '    install-gpu-deps - Install GPU-enabled Python packages'
	@echo ''
	@echo '🏃 Running the Application:'
	@echo '    run              - Run T-Bot trading system'
	@echo '    run-mock         - Run T-Bot in mock mode (no API keys)'
	@echo '    run-web          - Start web interface'
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

setup: ## Complete setup (venv, deps, external libs, GPU)
	@echo "🔧 Complete T-Bot Setup..."
	@echo "📋 Running pre-installation checks..."
	@bash $(SCRIPTS_DIR)/setup/pre_install.sh
	@$(MAKE) -s setup-venv
	@$(MAKE) -s setup-external
	@$(MAKE) -s install-deps
	@$(MAKE) -s setup-gpu
	@$(MAKE) -s install-gpu-deps
	@$(MAKE) -s services-up
	@$(MAKE) -s migrate
	@echo "✅ Complete setup finished!"
	@echo "ℹ️  Run 'make run' to start the application"
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
	@bash $(SCRIPTS_DIR)/setup/external_libs.sh install
	@echo "✅ External libraries installed!"

setup-gpu: ## Install GPU/CUDA dependencies
	@echo "🎮 Setting up GPU/CUDA support..."
	@bash $(SCRIPTS_DIR)/setup/cuda.sh install
	@bash $(SCRIPTS_DIR)/setup/cudnn.sh install
	@bash $(SCRIPTS_DIR)/setup/lightgbm.sh install
	@echo "✅ GPU setup completed!"

install-deps: ## Install Python dependencies
	@echo "📦 Installing Python dependencies..."
	@cd $(PROJECT_DIR) && bash $(SCRIPTS_DIR)/setup/install_requirements.sh
	@echo "✅ Python dependencies installed!"

install-gpu-deps: ## Install GPU-enabled Python packages
	@echo "🎮 Installing GPU-enabled Python packages..."
	@$(PIP) install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@$(PIP) install --upgrade tensorflow[and-cuda]
	@$(PIP) install --upgrade cupy-cuda12x
	@$(PIP) install --upgrade rapids-cuda12
	@echo "✅ GPU-enabled packages installed!"

# ============================================================================
# Running Commands
# ============================================================================

run: ## Run T-Bot trading system
	@echo "🚀 Starting T-Bot Trading System..."
	@echo "⚠️  Checking services..."
	@$(MAKE) services-check
	@source $(VENV)/bin/activate && python -m src.main

run-mock: ## Run T-Bot in mock mode (no API keys)
	@echo "🚀 Starting T-Bot in Mock Mode..."
	@source $(VENV)/bin/activate && MOCK_MODE=true python -m src.main

run-web: ## Start web interface
	@echo "🌐 Starting Web Interface..."
	@source $(VENV)/bin/activate && python -m src.web_interface.app

migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	@source $(VENV)/bin/activate && alembic upgrade head
	@echo "✅ Migrations completed!"

# ============================================================================
# Docker & Services Commands
# ============================================================================

docker-build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	@docker build -t tbot-backend:latest -f $(DOCKER_DIR)/Dockerfile.backend .
	@docker build -t tbot-gpu:latest -f $(DOCKER_DIR)/Dockerfile.gpu .
	@echo "✅ Docker images built!"

docker-up: ## Start all services with Docker
	@echo "🐳 Starting all services with Docker..."
	@$(DOCKER_COMPOSE_FULL) up -d
	@echo "✅ All services started!"

docker-down: ## Stop Docker services
	@echo "🛑 Stopping Docker services..."
	@$(DOCKER_COMPOSE_FULL) down
	@echo "✅ Services stopped!"

docker-logs: ## Show Docker logs
	@$(DOCKER_COMPOSE_FULL) logs -f

docker-clean: ## Clean Docker resources
	@echo "🧹 Cleaning Docker resources..."
	@$(DOCKER_COMPOSE_FULL) down -v
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
	@docker ps | grep -q tbot-postgresql || (echo "⚠️  PostgreSQL not running. Run 'make services-up'" && exit 1)
	@docker ps | grep -q tbot-redis || (echo "⚠️  Redis not running. Run 'make services-up'" && exit 1)
	@echo "✅ All services are running!"

# ============================================================================
# Testing & Quality Commands
# ============================================================================

test: ## Run all tests
	@echo "🧪 Running all tests..."
	@source $(VENV)/bin/activate && python -m pytest $(TEST_DIR)/ -v --tb=short
	@echo "✅ Tests completed!"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@source $(VENV)/bin/activate && python -m pytest $(TEST_DIR)/unit/ -v --tb=short
	@echo "✅ Unit tests completed!"

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@source $(VENV)/bin/activate && python -m pytest $(TEST_DIR)/integration/ -v --tb=short
	@echo "✅ Integration tests completed!"

test-mock: ## Run tests in mock mode
	@echo "🧪 Running tests in mock mode..."
	@source $(VENV)/bin/activate && MOCK_MODE=true python -m pytest $(TEST_DIR)/ -v --tb=short
	@echo "✅ Mock tests completed!"

coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	@source $(VENV)/bin/activate && python -m pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generated!"
	@echo "📁 HTML report: htmlcov/index.html"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@source $(VENV)/bin/activate && ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Linting completed!"

format: ## Format code with ruff and black
	@echo "🎨 Formatting code..."
	@source $(VENV)/bin/activate && ruff check $(SRC_DIR) $(TEST_DIR) --fix
	@source $(VENV)/bin/activate && ruff format $(SRC_DIR) $(TEST_DIR)
	@source $(VENV)/bin/activate && black $(SRC_DIR) $(TEST_DIR) --line-length 100
	@echo "✅ Code formatted!"

typecheck: ## Run type checking with mypy
	@echo "🔍 Running type checking..."
	@source $(VENV)/bin/activate && mypy $(SRC_DIR) --ignore-missing-imports
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
	@bash $(SCRIPTS_DIR)/validate_setup.sh
	@echo "✅ Validation completed!"

audit: ## Complete system audit
	@echo "🔍 Running complete system audit..."
	@bash $(SCRIPTS_DIR)/final_validation.sh
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