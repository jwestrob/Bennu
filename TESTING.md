# Testing Guide

This document describes the comprehensive testing system for the microbial genome knowledge graph project.

## Quick Start

```bash
# Show all available tests
python run_tests.py --discover

# Run all tests
python run_tests.py

# Run quick smoke tests
python run_tests.py --smoke

# Run with coverage
python run_tests.py --coverage

# Run specific module tests
python run_tests.py --module ingest

# Run by marker
python run_tests.py --marker unit
```

## Master Test Runner

The `run_tests.py` script provides a comprehensive test execution system with automatic test discovery. It requires **no manual updates** when you add new tests - just follow the existing naming conventions.

### Key Features

- **Automatic Discovery**: Finds all tests following pytest conventions
- **Multiple Execution Modes**: Run all, by module, by marker, smoke tests, etc.
- **Coverage Integration**: Built-in coverage reporting
- **HTML Reports**: Generates detailed test reports
- **CI/CD Ready**: Optimized modes for continuous integration
- **Parallel Execution**: Support for running tests in parallel

## Test Organization

### Directory Structure

```
src/tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_*.py               # Standalone test files
├── test_ingest/            # Ingest module tests
│   ├── test_00_prepare_inputs.py
│   ├── test_01_quast.py
│   └── test_03_prodigal.py
├── test_build_kg/          # Knowledge graph tests
├── test_llm/               # LLM/RAG tests
└── test_integration/       # Integration tests
```

### Test Categories (Markers)

Tests are organized using pytest markers defined in `pytest.ini`:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests involving multiple components
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.external` - Tests requiring external tools/services

## Usage Examples

### Basic Usage

```bash
# Run all tests with verbose output
python run_tests.py

# Run all tests quietly
python run_tests.py --quiet

# Run with coverage analysis
python run_tests.py --coverage

# Run tests in parallel (faster execution)
python run_tests.py --parallel
```

### Discovery and Dry Runs

```bash
# See what tests are available
python run_tests.py --discover

# See what would be run without executing
python run_tests.py --dry-run

# See what unit tests would be run
python run_tests.py --dry-run --marker unit
```

### Targeted Testing

```bash
# Run only unit tests
python run_tests.py --marker unit

# Run only integration tests
python run_tests.py --marker integration

# Run tests for specific module
python run_tests.py --module ingest

# Run tests for knowledge graph module
python run_tests.py --module build_kg
```

### Development Workflows

```bash
# Quick validation during development
python run_tests.py --smoke

# Full validation before commit
python run_tests.py --full

# CI/CD optimized run
python run_tests.py --ci
```

### Shell Wrapper

For convenience, use the shell wrapper:

```bash
# All the same commands work with the shell wrapper
./test.sh --discover
./test.sh --smoke
./test.sh --module ingest --coverage
```

## Test Execution Modes

### 1. All Tests (Default)
```bash
python run_tests.py
```
- Runs all discovered tests
- Generates HTML report
- Verbose output by default

### 2. Smoke Tests
```bash
python run_tests.py --smoke
```
- Quick validation (unit tests only)
- Excludes slow and external tests  
- Stops on first failure (`-x`)
- Ideal for rapid development feedback

### 3. Full Validation
```bash
python run_tests.py --full
```
- Comprehensive test suite
- Includes coverage analysis (80% threshold)
- Detailed HTML and XML reports
- Suitable for pre-release validation

### 4. CI/CD Mode
```bash
python run_tests.py --ci
```
- Optimized for continuous integration
- JUnit XML output for CI systems
- Coverage reporting with 70% threshold
- Fails fast (max 5 failures)
- Strict marker validation

### 5. Module-Specific
```bash
python run_tests.py --module <module_name>
```
Available modules (auto-discovered):
- `ingest` - Data ingestion pipeline tests
- `build_kg` - Knowledge graph construction tests  
- `llm` - LLM and RAG system tests
- `integration` - End-to-end integration tests
- `standalone` - Standalone test files

### 6. Marker-Based
```bash
python run_tests.py --marker <marker_name>
```
Available markers:
- `unit` - Fast, isolated tests
- `integration` - Multi-component tests
- `slow` - Time-intensive tests
- `external` - Tests requiring external dependencies

## Reports and Output

### Test Reports
Generated in `test_reports/` directory:
- `all_tests.html` - Complete test results
- `<module>_tests.html` - Module-specific results
- `<marker>_tests.html` - Marker-specific results
- `junit.xml` - CI/CD compatible results

### Coverage Reports
Generated in `coverage/` directory:
- `html/` - Interactive HTML coverage report
- `coverage.xml` - XML coverage for CI/CD
- Terminal output shows coverage summary

## Adding New Tests

The system automatically discovers new tests when you:

1. **Add test files** following the `test_*.py` pattern
2. **Add test directories** following the `test_<module>/` pattern  
3. **Add test functions** following the `test_*` pattern
4. **Use existing markers** to categorize tests

### Example: Adding a New Test Module

```bash
# Create new test directory
mkdir src/tests/test_new_module

# Add test file
cat > src/tests/test_new_module/test_feature.py << 'EOF'
import pytest

@pytest.mark.unit
def test_basic_functionality():
    assert True

@pytest.mark.integration  
def test_integration_scenario():
    assert True
EOF
```

The new tests will be automatically discovered:

```bash
python run_tests.py --discover  # Shows new module
python run_tests.py --module new_module  # Runs new tests
```

## Advanced Usage

### Custom pytest Arguments

Pass additional arguments directly to pytest:

```bash
# Run with custom pytest options
python run_tests.py --verbose --tb=long

# Run specific test file
python run_tests.py src/tests/test_ingest/test_prodigal.py

# Run with keyword filtering
python run_tests.py -k "test_prepare"

# Run with multiple markers
python run_tests.py -m "unit and not slow"
```

### Environment Variables

Set environment variables for test configuration:

```bash
# Set custom test database
export TEST_DB_URL="postgresql://test:test@localhost/test_db"
python run_tests.py

# Enable debug mode
export DEBUG=1
python run_tests.py --marker integration
```

## Troubleshooting

### Common Issues

1. **No tests found**: Ensure test files follow `test_*.py` naming convention
2. **Import errors**: Check that `src/` is in your Python path
3. **Missing dependencies**: Install test dependencies with `pip install -r requirements-test.txt`
4. **Coverage issues**: Ensure source code is in `src/` directory

### Debug Mode

```bash
# Run with maximum verbosity
python run_tests.py --verbose -s

# Run single test with debugging
python run_tests.py -k "test_specific_function" -s --tb=long

# Show test collection without running
python run_tests.py --collect-only
```

## Integration with IDEs

### VS Code
- Use the Python extension's test discovery
- Run individual tests from the editor
- Set breakpoints for debugging

### PyCharm
- Configure pytest as the test runner
- Use the integrated test runner interface
- Debug tests with the built-in debugger

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - run: pip install -r requirements.txt
    - run: python run_tests.py --ci
    - uses: codecov/codecov-action@v3
      with:
        file: ./coverage/coverage.xml
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'python run_tests.py --ci'
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test_reports',
                        reportFiles: '*.html',
                        reportName: 'Test Results'
                    ])
                    publishTestResults(
                        testResultsPattern: 'test_reports/junit.xml'
                    )
                }
            }
        }
    }
}
```

## Best Practices

1. **Write tests for new features** before implementing them (TDD)
2. **Use appropriate markers** to categorize tests properly  
3. **Keep unit tests fast** - avoid external dependencies
4. **Use fixtures** from `conftest.py` for common test data
5. **Run smoke tests frequently** during development
6. **Run full validation** before commits/releases
7. **Check coverage reports** to identify untested code
8. **Use descriptive test names** that explain what they test

## Performance

### Parallel Execution

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
python run_tests.py --parallel
```

### Test Selection

```bash
# Run only fast tests during development
python run_tests.py --marker "unit and not slow"

# Skip external dependencies
python run_tests.py --marker "not external"

# Run changed tests only (requires pytest-testmon)
python run_tests.py --testmon
```

This testing system is designed to grow with your project while maintaining zero maintenance overhead for test discovery and execution.
