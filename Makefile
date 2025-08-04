# Trading Bot Makefile
# Simple commands for database migrations and testing

SHELL := /bin/bash

.PHONY: migrate test docker-up host-ip

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

# Start database services with Docker
docker-up:
	@echo "🐳 Starting database services with Docker..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Database services started!"
	@echo "📊 PostgreSQL: localhost:5432"
	@echo "🔴 Redis: localhost:6379" 

host-ip:
	@(ip route show default | awk '{print $3}')