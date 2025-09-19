#!/bin/bash

# T-Bot Unit Test Runner - Module by Module
# This script runs unit tests one module at a time to avoid contamination issues

set -e  # Exit on any error

# Configuration
TIMEOUT=600  # 10 minutes per module
TEST_DIR="tests/unit"
LOG_DIR="logs/unit_tests"
SUMMARY_FILE="test_summary.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize
echo -e "${BLUE}T-Bot Unit Test Runner - Module by Module${NC}"
echo "Starting at: $(date)"
echo "============================================"

# Create log directory
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log
rm -f "$SUMMARY_FILE"

# Initialize counters
total_modules=0
passed_modules=0
failed_modules=0
total_tests=0
passed_tests=0
failed_tests=0

# Function to run tests for a module
run_module_tests() {
    local module_path=$1
    local module_name=$(basename "$module_path")
    local log_file="$LOG_DIR/${module_name}.log"

    echo -e "${YELLOW}Testing module: ${module_name}${NC}"
    echo "Module: $module_name" >> "$SUMMARY_FILE"

    # Record start time
    local start_time=$(date +%s)

    # Run tests with timeout
    if timeout $TIMEOUT python -m pytest "$module_path" -v --tb=short > "$log_file" 2>&1; then
        # Calculate duration
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        # Extract test counts from log
        local module_tests=$(grep -E "^.*::.*PASSED|^.*::.*FAILED" "$log_file" | wc -l)
        local module_passed
        local module_failed

        # Get passed count safely
        module_passed=$(grep -c "PASSED" "$log_file" 2>/dev/null || echo "0")
        module_passed=$(echo "$module_passed" | head -n1 | tr -d '\n\r')

        # Get failed count safely
        module_failed=$(grep -c "FAILED" "$log_file" 2>/dev/null || echo "0")
        module_failed=$(echo "$module_failed" | head -n1 | tr -d '\n\r')

        # Format timing display
        local time_display
        if [ "$minutes" -gt 0 ]; then
            time_display="(took ${minutes}:$(printf "%02d" $seconds) minutes)"
        else
            time_display="(took ${seconds}s)"
        fi

        if [ "$module_failed" -eq 0 ]; then
            echo -e "${GREEN}✓ PASSED: ${module_name} (${module_tests} tests) - ${time_display}${NC}"
            echo "  Status: PASSED (${module_tests} tests) - ${time_display}" >> "$SUMMARY_FILE"
            passed_modules=$((passed_modules + 1))
        else
            echo -e "${RED}✗ FAILED: ${module_name} (${module_failed}/${module_tests} failures) - ${time_display}${NC}"
            echo "  Status: FAILED (${module_failed}/${module_tests} failures) - ${time_display}" >> "$SUMMARY_FILE"
            echo "  See: $log_file" >> "$SUMMARY_FILE"
            failed_modules=$((failed_modules + 1))
        fi

        total_tests=$((total_tests + module_tests))
        passed_tests=$((passed_tests + module_passed))
        failed_tests=$((failed_tests + module_failed))

    else
        echo -e "${RED}✗ TIMEOUT/ERROR: ${module_name}${NC}"
        echo "  Status: TIMEOUT/ERROR" >> "$SUMMARY_FILE"
        echo "  See: $log_file" >> "$SUMMARY_FILE"
        failed_modules=$((failed_modules + 1))
    fi

    total_modules=$((total_modules + 1))
    echo "" >> "$SUMMARY_FILE"
}

# Get all test modules (directories and individual test files)
test_modules=()

# First, add individual test files in the root of tests/unit/
for test_file in "$TEST_DIR"/test_*.py; do
    if [ -f "$test_file" ]; then
        test_modules+=("$test_file")
    fi
done

# Then, add test module directories
for test_dir in "$TEST_DIR"/test_*/; do
    if [ -d "$test_dir" ]; then
        test_modules+=("$test_dir")
    fi
done

# Sort modules for consistent execution order
IFS=$'\n' test_modules=($(sort <<<"${test_modules[*]}"))
unset IFS

echo "Found ${#test_modules[@]} test modules to run"
echo ""

# Run tests for each module
for module in "${test_modules[@]}"; do
    run_module_tests "$module"
done

# Final summary
echo "============================================"
echo -e "${BLUE}Test Execution Summary${NC}"
echo "Completed at: $(date)"
echo ""

echo "Modules:"
echo "  Total: $total_modules"
echo -e "  Passed: ${GREEN}$passed_modules${NC}"
echo -e "  Failed: ${RED}$failed_modules${NC}"
echo ""

echo "Tests:"
echo "  Total: $total_tests"
echo -e "  Passed: ${GREEN}$passed_tests${NC}"
echo -e "  Failed: ${RED}$failed_tests${NC}"
echo ""

# Write summary to file
{
    echo "============================================"
    echo "FINAL SUMMARY"
    echo "============================================"
    echo "Execution completed at: $(date)"
    echo ""
    echo "Modules: $passed_modules/$total_modules passed"
    echo "Tests: $passed_tests/$total_tests passed"
    echo ""
    if [ $failed_modules -eq 0 ]; then
        echo "SUCCESS: All modules passed!"
    else
        echo "FAILURES: $failed_modules modules failed"
    fi
} >> "$SUMMARY_FILE"

# Exit with appropriate code
if [ $failed_modules -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: All modules passed!${NC}"
    exit 0
else
    echo -e "${RED}FAILURES: $failed_modules modules failed${NC}"
    echo "Check individual log files in $LOG_DIR/ for details"
    echo "Full summary available in $SUMMARY_FILE"
    exit 1
fi