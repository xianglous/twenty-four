#!/bin/bash

# Function to run a command and check its status
run_command() {
    local description=$1
    shift
    local command=("$@")
    
    echo -e "\n=== Running $description ==="
    if "${command[@]}"; then
        echo "✓ $description completed successfully"
        return 0
    else
        echo "✗ $description failed"
        return 1
    fi
}

# Get the directory to format (default to current directory)
TARGET_DIR=${1:-.}

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Run isort
run_command "isort (import sorting)" isort --profile black "$TARGET_DIR"
ISORT_STATUS=$?

# Run black
run_command "black (code formatting)" black "$TARGET_DIR"
BLACK_STATUS=$?

# Run flake8
run_command "flake8 (linting)" flake8 "$TARGET_DIR"
FLAKE8_STATUS=$?

# Print summary
echo -e "\n=== Formatting Summary ==="
echo "isort: $(if [ $ISORT_STATUS -eq 0 ]; then echo "✓"; else echo "✗"; fi)"
echo "black: $(if [ $BLACK_STATUS -eq 0 ]; then echo "✓"; else echo "✗"; fi)"
echo "flake8: $(if [ $FLAKE8_STATUS -eq 0 ]; then echo "✓"; else echo "✗"; fi)"

# Exit with error if any tool failed
if [ $ISORT_STATUS -ne 0 ] || [ $BLACK_STATUS -ne 0 ] || [ $FLAKE8_STATUS -ne 0 ]; then
    exit 1
fi 