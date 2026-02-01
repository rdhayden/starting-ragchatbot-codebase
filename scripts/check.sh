#!/bin/bash
# Run all code quality checks

set -e

cd "$(dirname "$0")/.."

echo "=== Code Quality Checks ==="
echo

echo "Checking code formatting with black..."
uv run black --check .
echo "Formatting check passed!"
echo

echo "=== All checks passed! ==="
