#!/usr/bin/env python3
"""
Master test runner for the microbial genome knowledge graph project.

This script automatically discovers and runs all test cases with various execution modes.
No manual updates needed when adding new tests - just follow existing naming conventions.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time
from datetime import datetime


class TestRunner:
    """Comprehensive test runner with automatic test discovery."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "src" / "tests"
        self.coverage_dir = self.project_root / "coverage"
        self.reports_dir = self.project_root / "test_reports"
        
        # Ensure output directories exist
        self.coverage_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test markers from pytest.ini
        self.markers = {
            'unit': 'Unit tests (fast, isolated)',
            'integration': 'Integration tests (multiple components)',
            'slow': 'Slow tests (may take significant time)',
            'external': 'Tests requiring external tools'
        }
        
        # Auto-discover test modules
        self.modules = self._discover_test_modules()
        
        # Check for optional plugins
        self.has_coverage = self._check_plugin_availability('--cov')
        self.has_html = self._check_plugin_availability('--html')
        self.has_xdist = self._check_plugin_availability('-n')
    
    def _discover_test_modules(self) -> Dict[str, Path]:
        """Automatically discover test modules from directory structure."""
        modules = {}
        
        if not self.test_dir.exists():
            print(f"Warning: Test directory {self.test_dir} not found")
            return modules
        
        # Scan for test directories
        for item in self.test_dir.iterdir():
            if item.is_dir() and item.name.startswith('test_'):
                module_name = item.name[5:]  # Remove 'test_' prefix
                modules[module_name] = item
        
        # Add standalone test files as 'standalone' module
        standalone_tests = [
            f for f in self.test_dir.iterdir() 
            if f.is_file() and f.name.startswith('test_') and f.suffix == '.py'
        ]
        if standalone_tests:
            modules['standalone'] = self.test_dir
        
        return modules
    
    def _get_test_count(self, pytest_args: List[str]) -> int:
        """Get count of tests that would be run with given arguments."""
        try:
            cmd = ['python', '-m', 'pytest', '--collect-only', '-q'] + pytest_args
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'test' in line.lower() and ('collected' in line or 'selected' in line):
                        # Extract number from lines like "collected 15 items"
                        words = line.split()
                        for i, word in enumerate(words):
                            if word.isdigit():
                                return int(word)
            return 0
        except Exception:
            return 0
    
    def _check_plugin_availability(self, option: str) -> bool:
        """Check if a pytest option/plugin is available."""
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--help'],
                capture_output=True, text=True, cwd=self.project_root
            )
            return option in result.stdout
        except Exception:
            return False
    
    def _run_pytest(self, args: List[str], description: str) -> Tuple[int, float]:
        """Run pytest with given arguments and return (exit_code, duration)."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: python -m pytest {' '.join(args)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = subprocess.run(
            ['python', '-m', 'pytest'] + args,
            cwd=self.project_root
        )
        duration = time.time() - start_time
        
        return result.returncode, duration
    
    def show_discovery(self):
        """Show discovered tests and modules."""
        print("\n" + "="*60)
        print("TEST DISCOVERY SUMMARY")
        print("="*60)
        
        print(f"Test directory: {self.test_dir}")
        print(f"Discovered modules: {len(self.modules)}")
        
        total_tests = 0
        for module_name, module_path in self.modules.items():
            if module_name == 'standalone':
                test_pattern = [str(self.test_dir / "test_*.py")]
            else:
                test_pattern = [str(module_path)]
            
            count = self._get_test_count(test_pattern)
            total_tests += count
            print(f"  {module_name:15} | {count:3} tests | {module_path}")
        
        print(f"\nTotal tests found: {total_tests}")
        
        # Show available markers
        print(f"\nAvailable markers:")
        for marker, desc in self.markers.items():
            count = self._get_test_count(['-m', marker])
            print(f"  {marker:12} | {count:3} tests | {desc}")
        
        # Show plugin availability
        print(f"\nAvailable plugins:")
        plugins = [
            ('coverage (pytest-cov)', self.has_coverage),
            ('HTML reports (pytest-html)', self.has_html),
            ('parallel execution (pytest-xdist)', self.has_xdist)
        ]
        for plugin_name, available in plugins:
            status = "✓" if available else "✗"
            print(f"  {status} {plugin_name}")
        
        print("\n" + "="*60)
    
    def run_all(self, coverage: bool = False, parallel: bool = False, verbose: bool = True):
        """Run all tests."""
        args = []
        
        if coverage and self.has_coverage:
            args.extend([
                '--cov=src',
                '--cov-report=term-missing'
            ])
            if self.coverage_dir.exists():
                args.append(f'--cov-report=html:{self.coverage_dir / "html"}')
        elif coverage and not self.has_coverage:
            print("Warning: Coverage requested but pytest-cov not available. Install with: pip install pytest-cov")
        
        if parallel and self.has_xdist:
            args.extend(['-n', 'auto'])
        elif parallel and not self.has_xdist:
            print("Warning: Parallel execution requested but pytest-xdist not available. Install with: pip install pytest-xdist")
        
        if verbose:
            args.append('-v')
        
        return self._run_pytest(args, "All Tests")
    
    def run_by_marker(self, marker: str, coverage: bool = False):
        """Run tests with specific marker."""
        if marker not in self.markers:
            print(f"Error: Unknown marker '{marker}'. Available: {list(self.markers.keys())}")
            return 1, 0
        
        args = ['-m', marker]
        
        if coverage and self.has_coverage:
            args.extend([
                '--cov=src',
                '--cov-report=term-missing'
            ])
        elif coverage and not self.has_coverage:
            print("Warning: Coverage requested but pytest-cov not available. Install with: pip install pytest-cov")
        
        return self._run_pytest(args, f"Tests with marker: {marker}")
    
    def run_by_module(self, module: str, coverage: bool = False):
        """Run tests for specific module."""
        if module not in self.modules:
            print(f"Error: Unknown module '{module}'. Available: {list(self.modules.keys())}")
            return 1, 0
        
        module_path = self.modules[module]
        
        if module == 'standalone':
            test_pattern = str(self.test_dir / "test_*.py")
        else:
            test_pattern = str(module_path)
        
        args = [test_pattern]
        
        if coverage and self.has_coverage:
            args.extend([
                '--cov=src',
                '--cov-report=term-missing'
            ])
        elif coverage and not self.has_coverage:
            print("Warning: Coverage requested but pytest-cov not available. Install with: pip install pytest-cov")
        
        return self._run_pytest(args, f"Module: {module}")
    
    def run_smoke_tests(self):
        """Run quick smoke tests (unit tests, not slow or external)."""
        args = ['-x']  # Stop on first failure for quick feedback
        
        # Try to run unit tests that aren't slow/external
        marker_count = self._get_test_count(['-m', 'unit and not slow and not external'])
        if marker_count > 0:
            args.extend(['-m', 'unit and not slow and not external'])
            description = "Smoke Tests (Unit tests, fast)"
        else:
            # Fallback: just run a few tests quickly
            args.extend(['--maxfail=3', '-q'])
            description = "Smoke Tests (Quick validation)"
        
        return self._run_pytest(args, description)
    
    def run_full_validation(self, coverage: bool = True):
        """Run comprehensive validation suite."""
        args = ['--tb=short']
        
        if coverage and self.has_coverage:
            args.extend([
                '--cov=src',
                '--cov-report=term-missing',
                '--cov-fail-under=70'
            ])
            if self.coverage_dir.exists():
                args.append(f'--cov-report=html:{self.coverage_dir / "html"}')
        elif coverage and not self.has_coverage:
            print("Warning: Coverage requested but pytest-cov not available. Install with: pip install pytest-cov")
        
        return self._run_pytest(args, "Full Validation Suite")
    
    def run_ci_tests(self):
        """Run tests optimized for CI/CD."""
        args = [
            '--tb=short',
            '--strict-markers',
            '--maxfail=5'
        ]
        
        if self.has_coverage:
            args.extend([
                '--cov=src',
                '--cov-report=term',
                '--cov-fail-under=60'
            ])
        
        return self._run_pytest(args, "CI/CD Test Suite")
    
    def run_basic(self):
        """Run basic tests without any optional features."""
        args = ['-v']
        return self._run_pytest(args, "Basic Test Run")
    
    def dry_run(self, *pytest_args):
        """Show what tests would be run without executing them."""
        args = ['--collect-only', '-q'] + list(pytest_args)
        print(f"Dry run - would execute: python -m pytest {' '.join(args)}")
        return self._run_pytest(args, "Dry Run")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Master test runner for microbial genome knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                          # Run all tests
  python run_tests.py --discover              # Show available tests
  python run_tests.py --basic                 # Basic test run (no plugins)
  python run_tests.py --smoke                 # Quick smoke tests
  python run_tests.py --marker unit           # Run unit tests only
  python run_tests.py --module ingest         # Run ingest tests only
  python run_tests.py --full                  # Full validation suite
  python run_tests.py --ci                    # CI/CD optimized run
  python run_tests.py --coverage              # Run with coverage (if available)
  python run_tests.py --parallel              # Run in parallel (if available)
  python run_tests.py --dry-run               # Show what would run
        """
    )
    
    # Test execution modes
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        '--discover', action='store_true',
        help='Show discovered tests and exit'
    )
    execution_group.add_argument(
        '--basic', action='store_true',
        help='Run basic tests without optional plugins'
    )
    execution_group.add_argument(
        '--smoke', action='store_true',
        help='Run quick smoke tests'
    )
    execution_group.add_argument(
        '--full', action='store_true',
        help='Run full validation suite'
    )
    execution_group.add_argument(
        '--ci', action='store_true',
        help='Run CI/CD optimized tests'
    )
    execution_group.add_argument(
        '--marker', choices=['unit', 'integration', 'slow', 'external'],
        help='Run tests with specific marker'
    )
    execution_group.add_argument(
        '--module', 
        help='Run tests for specific module (discovered automatically)'
    )
    execution_group.add_argument(
        '--dry-run', action='store_true',
        help='Show what tests would be run without executing'
    )
    
    # Test options
    parser.add_argument(
        '--coverage', action='store_true',
        help='Generate coverage report (requires pytest-cov)'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    # Pass-through arguments to pytest
    parser.add_argument(
        'pytest_args', nargs='*',
        help='Additional arguments passed to pytest'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Handle discovery mode
    if args.discover:
        runner.show_discovery()
        return 0
    
    # Handle dry run
    if args.dry_run:
        exit_code, _ = runner.dry_run(*args.pytest_args)
        return exit_code
    
    # Track execution time
    start_time = time.time()
    
    # Execute tests based on mode
    if args.basic:
        exit_code, duration = runner.run_basic()
    elif args.smoke:
        exit_code, duration = runner.run_smoke_tests()
    elif args.full:
        exit_code, duration = runner.run_full_validation(coverage=args.coverage)
    elif args.ci:
        exit_code, duration = runner.run_ci_tests()
    elif args.marker:
        exit_code, duration = runner.run_by_marker(args.marker, coverage=args.coverage)
    elif args.module:
        exit_code, duration = runner.run_by_module(args.module, coverage=args.coverage)
    else:
        # Default: run all tests
        exit_code, duration = runner.run_all(
            coverage=args.coverage,
            parallel=args.parallel,
            verbose=not args.quiet
        )
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Status: {'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"Exit code: {exit_code}")
    print(f"Duration: {duration:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if runner.reports_dir.exists():
        print(f"Reports directory: {runner.reports_dir}")
    
    if args.coverage and runner.coverage_dir.exists():
        print(f"Coverage reports: {runner.coverage_dir}")
    
    print(f"{'='*60}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
