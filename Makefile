# Trading Bot Makefile
# Simple commands for database migrations and testing

# Use bash for Makefile recipes
SHELL := /bin/bash

.PHONY: migrate test test-unit test-integration coverage docker-up host-ip format wsl-test wsl-test-unit wsl-test-integration wsl-coverage

# Run database migrations
migrate:
	@echo "🔄 Running database migrations..."
	source ~/.venv/bin/activate && alembic upgrade head
	@echo "✅ Migrations completed!"

# Run all tests with proper setup
test:
	@echo "🧪 Running all tests..."
	source ~/.venv/bin/activate && python -m pytest tests/ -v --tb=short
	@echo "✅ Tests completed!"

test-unit:
	@echo "🧪 Running all unit tests..."
	source ~/.venv/bin/activate && python -m pytest tests/unit/ -v --tb=short
	@echo "✅ Tests completed!"
	
test-integration:
	@echo "🧪 Running all integration tests..."
	source ~/.venv/bin/activate && python -m pytest tests/integration/ -v --tb=short
	@echo "✅ Tests completed!"

coverage:
	@echo "🧪 Running coverage..."
	source ~/.venv/bin/activate && python -m pytest tests/ --cov=src --cov-report=term-missing
	@echo "✅ Coverage completed!"

# WSL wrappers to ensure correct environment and path quoting
wsl-test:
	@echo "🧪 (WSL) Running all tests..."
	wsl -- bash -lc "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && python -m pytest tests/ -v --tb=short | cat"

wsl-test-unit:
	@echo "🧪 (WSL) Running unit tests..."
	wsl -- bash -lc "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && python -m pytest tests/unit/ -v --tb=short | cat"

wsl-test-integration:
	@echo "🧪 (WSL) Running integration tests..."
	wsl -- bash -lc "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && python -m pytest tests/integration/ -v --tb=short | cat"

wsl-coverage:
	@echo "🧪 (WSL) Running coverage..."
	wsl -- bash -lc "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && python -m pytest tests/ --cov=src --cov-report=term-missing | cat"

# Format all Python files using autopep8
format:
	@echo "🎨 Formatting all Python files with autopep8..."
	python -m autopep8 --in-place --recursive --aggressive --aggressive src/ tests/ --max-line-length=100
	@echo "✅ Formatting completed!"

# Start database services with Docker
docker-up:
	@echo "🐳 Starting database services with Docker..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Database services started!"
	@echo "📊 PostgreSQL: localhost:5432"
	@echo "🔴 Redis: localhost:6379" 

host-ip:
	@(ip route show default | awk '{print $3}')