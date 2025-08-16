#!/bin/bash

# Test Docker build and identify issues

echo "Testing Docker build for T-Bot Trading System..."
echo "========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Test building the backend service
echo ""
echo "Testing backend build..."
docker-compose build backend --no-cache 2>&1 | tail -20

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Backend build successful"
else
    echo "❌ Backend build failed"
fi

echo ""
echo "Checking for missing files or dependencies..."
# Check critical files
files_to_check=(
    "Dockerfile"
    "requirements.txt"
    "src/main.py"
    "docker-compose.yml"
    ".env"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
    fi
done

echo ""
echo "Validating docker-compose configuration..."
docker-compose config --quiet
if [ $? -eq 0 ]; then
    echo "✅ docker-compose.yml is valid"
else
    echo "❌ docker-compose.yml has errors"
fi