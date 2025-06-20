#!/bin/bash
# Simple wrapper for the master test runner

set -e

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Run the Python test runner with all arguments
python run_tests.py "$@"
