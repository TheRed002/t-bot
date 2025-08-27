#!/bin/bash
# Script to check test coverage for T-Bot Trading System

echo "========================================="
echo "T-Bot Trading System - Test Coverage Check"
echo "========================================="

# Activate virtual environment
source ~/.venv/bin/activate

# Change to project directory
cd "/mnt/e/Work/P-41 Trading/code/t-bot"

# Run tests with coverage
echo "Running test suite with coverage..."
python -m pytest tests/ \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=json \
    -v \
    --tb=short \
    2>&1 | tee test_results.txt

# Extract coverage percentage
COVERAGE=$(grep "TOTAL" test_results.txt | awk '{print $NF}' | sed 's/%//')

echo ""
echo "========================================="
echo "Coverage Summary:"
echo "========================================="
grep "TOTAL" test_results.txt

# Check if coverage meets requirement
if [ -z "$COVERAGE" ]; then
    echo "ERROR: Could not determine coverage percentage"
    exit 1
fi

# Convert to integer for comparison
COVERAGE_INT=$(echo $COVERAGE | cut -d. -f1)

if [ "$COVERAGE_INT" -lt 90 ]; then
    echo ""
    echo "WARNING: Coverage is below 90% requirement!"
    echo "Current coverage: ${COVERAGE}%"
    echo "Required coverage: 90%"
    echo ""
    echo "Top uncovered modules:"
    grep "src/" test_results.txt | sort -t' ' -k4 -n | head -10
else
    echo ""
    echo "SUCCESS: Coverage meets requirement!"
    echo "Current coverage: ${COVERAGE}%"
fi

echo ""
echo "HTML coverage report generated at: htmlcov/index.html"
echo "JSON coverage report generated at: coverage.json"