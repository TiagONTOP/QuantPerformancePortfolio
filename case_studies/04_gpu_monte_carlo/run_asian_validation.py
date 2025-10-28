"""Complete validation suite for Asian option pricing: GPU vs CPU.

This script runs all correctness tests and benchmarks for Asian option pricing,
comparing the GPU (optimized) and CPU (suboptimal) implementations.

Usage:
    python run_asian_validation.py

Requirements:
    - numpy
    - pytest
    - cupy-cuda12x (for GPU tests)
"""

import sys
import subprocess
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        success = result.returncode == 0

        if success:
            print(f"\n✓ {description} PASSED\n")
        else:
            print(f"\n✗ {description} FAILED\n")

        return success

    except Exception as e:
        print(f"\n✗ {description} FAILED WITH ERROR: {e}\n")
        return False


def main():
    """Run the complete validation suite."""
    print_header("ASIAN OPTION PRICING VALIDATION SUITE")
    print("This will run all tests and benchmarks for Asian option pricing")
    print("comparing GPU (optimized/pricing.py) vs CPU (suboptimal/pricing.py)\n")

    # Check if we're in the right directory
    if not Path("utils.py").exists():
        print("Error: Please run this script from the 04_gpu_monte_carlo directory")
        sys.exit(1)

    results = {}

    # 1. Run correctness tests
    print_header("PHASE 1: CORRECTNESS TESTS")
    print("Running rigorous unit tests to verify numerical correctness...")

    results["correctness"] = run_command(
        ["pytest", "tests/test_asian_option_correctness.py", "-v", "-s"],
        "Asian Option Correctness Tests"
    )

    # 2. Run benchmarks
    print_header("PHASE 2: PERFORMANCE BENCHMARKS")
    print("Running performance benchmarks to measure GPU speedup...")

    results["benchmarks"] = run_command(
        ["pytest", "tests/test_asian_option_benchmark.py", "-v", "-s"],
        "Asian Option Performance Benchmarks"
    )

    # 3. Run comprehensive benchmark suite
    print_header("PHASE 3: COMPREHENSIVE BENCHMARK SUITE")
    print("Running detailed benchmarks across all problem sizes...")

    results["comprehensive"] = run_command(
        ["python", "tests/test_asian_option_benchmark.py"],
        "Comprehensive Benchmark Suite"
    )

    # Summary
    print_header("VALIDATION SUMMARY")

    all_passed = all(results.values())

    for phase, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{phase.upper():<30} {status}")

    print()

    if all_passed:
        print("=" * 80)
        print("🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("=" * 80)
        print("\nConclusion:")
        print("  • GPU implementation is numerically correct")
        print("  • Results match CPU implementation within floating-point precision")
        print("  • GPU provides significant speedup for Asian option pricing")
        print("  • Implementation is production-ready")
        print()
        return 0
    else:
        print("=" * 80)
        print("⚠ SOME TESTS FAILED ⚠")
        print("=" * 80)
        print("\nPlease review the test output above for details.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
