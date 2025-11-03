"""Complete test and benchmark suite for GPU Monte Carlo simulation.

This script runs ALL tests and benchmarks, and saves results to a file.

Usage:
    python tests/run_all_tests_and_benchmarks.py

Output:
    - Console: Real-time test progress
    - tests/benchmark_results.txt: Detailed benchmark results
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
import io


def print_header(title: str, file=None):
    """Print a formatted header."""
    line = "=" * 80
    header = f"\n{line}\n{title:^80}\n{line}\n"
    print(header)
    if file:
        file.write(header + "\n")


def run_command_with_capture(cmd: list, description: str, output_file=None) -> tuple:
    """Run a command and return success status and output."""
    print(f"\nRunning: {description}")
    print(f"Command: {' '.join(cmd)}\n")

    if output_file:
        output_file.write(f"\n{'='*80}\n")
        output_file.write(f"Test: {description}\n")
        output_file.write(f"Command: {' '.join(cmd)}\n")
        output_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"{'='*80}\n\n")

    try:
        # Set PYTHONPATH to include current directory
        env = os.environ.copy()
        current_dir = Path.cwd()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{current_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(current_dir)

        # Use python -m pytest instead of pytest directly
        if cmd[0] == "pytest":
            cmd = [sys.executable, "-m", "pytest"] + cmd[1:]
        elif cmd[0] == "python":
            cmd = [sys.executable] + cmd[1:]

        # Run and capture output
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        # Print to console
        print(output)

        # Write to file if provided
        if output_file:
            output_file.write(output)
            output_file.write("\n")

        status = "[OK] PASSED" if success else "[FAIL] FAILED"
        print(f"\n{status}: {description}\n")

        if output_file:
            output_file.write(f"\n{status}: {description}\n")
            output_file.write(f"{'='*80}\n\n")

        return success, output

    except Exception as e:
        error_msg = f"[ERROR] {description} FAILED WITH ERROR: {e}"
        print(error_msg)
        if output_file:
            output_file.write(error_msg + "\n")
        return False, str(e)


def main():
    """Run the complete test and benchmark suite."""

    # Check if we're in the right directory
    if not Path("utils.py").exists():
        print("Error: Please run this script from the 04_gpu_monte_carlo directory")
        sys.exit(1)

    # Create output directory and file
    output_dir = Path("tests")
    output_file_path = output_dir / "benchmark_results.txt"

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Write header
        output_file.write("=" * 80 + "\n")
        output_file.write("GPU MONTE CARLO - COMPLETE TEST AND BENCHMARK RESULTS\n")
        output_file.write("=" * 80 + "\n")
        output_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 80 + "\n\n")

        print_header("COMPLETE TEST AND BENCHMARK SUITE", output_file)
        print("Running all tests and benchmarks for GPU Monte Carlo simulation")
        print(f"Results will be saved to: {output_file_path}\n")

        results = {}

        # ===== PHASE 1: CORRECTNESS TESTS =====
        print_header("PHASE 1: CORRECTNESS TESTS", output_file)

        # Test 1: GPU Correctness Tests
        results["gpu_correctness"], _ = run_command_with_capture(
            ["pytest", "tests/test_correctness_gpu.py", "-v"],
            "GPU Correctness Tests (test_correctness_gpu.py)",
            output_file
        )

        # Test 2: General Correctness Tests
        results["correctness"], _ = run_command_with_capture(
            ["pytest", "tests/test_correctness.py", "-v"],
            "General Correctness Tests (test_correctness.py)",
            output_file
        )

        # Test 3: Asian Option Correctness
        results["asian_correctness"], _ = run_command_with_capture(
            ["pytest", "tests/test_asian_option_correctness.py", "-v"],
            "Asian Option Correctness Tests",
            output_file
        )

        # ===== PHASE 2: BENCHMARK TESTS =====
        print_header("PHASE 2: BENCHMARK TESTS", output_file)

        # Benchmark 1: GPU Benchmarks
        results["gpu_benchmarks"], _ = run_command_with_capture(
            ["pytest", "tests/test_benchmark_gpu.py", "-v"],
            "GPU Benchmarks (test_benchmark_gpu.py)",
            output_file
        )

        # Benchmark 2: New Benchmarks
        results["new_benchmarks"], _ = run_command_with_capture(
            ["pytest", "tests/test_benchmark_new.py", "-v"],
            "New Benchmarks (test_benchmark_new.py)",
            output_file
        )

        # Benchmark 3: Asian Option Benchmarks
        results["asian_benchmarks"], _ = run_command_with_capture(
            ["pytest", "tests/test_asian_option_benchmark.py", "-v"],
            "Asian Option Benchmarks",
            output_file
        )

        # ===== SUMMARY =====
        print_header("TEST AND BENCHMARK SUMMARY", output_file)

        # Count results
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total - passed

        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 80)
        output_file.write("\nDetailed Results:\n")
        output_file.write("-" * 80 + "\n")

        for phase, success in results.items():
            status = "[OK] PASSED" if success else "[FAIL] FAILED"
            line = f"{phase.replace('_', ' ').title():<40} {status}"
            print(line)
            output_file.write(line + "\n")

        # Overall summary
        print("\n" + "=" * 80)
        print(f"TOTAL: {passed}/{total} test suites passed")
        print("=" * 80 + "\n")

        output_file.write("\n" + "=" * 80 + "\n")
        output_file.write(f"TOTAL: {passed}/{total} test suites passed\n")
        output_file.write("=" * 80 + "\n\n")

        if passed == total:
            message = "[SUCCESS] ALL TEST SUITES PASSED!"
            print("=" * 80)
            print(message)
            print("=" * 80)
            print("\nConclusions:")
            print("  - All correctness tests pass")
            print("  - GPU implementation is numerically correct")
            print("  - Benchmarks show significant GPU speedup")
            print("  - Implementation is production-ready")
            print(f"\nFull results saved to: {output_file_path}")
            print()

            output_file.write("=" * 80 + "\n")
            output_file.write(message + "\n")
            output_file.write("=" * 80 + "\n")
            return 0
        else:
            message = f"[WARNING] {failed}/{total} TEST SUITES FAILED"
            print("=" * 80)
            print(message)
            print("=" * 80)
            print(f"\nFull results saved to: {output_file_path}")
            print("Please review the output above for details.")
            print()

            output_file.write("=" * 80 + "\n")
            output_file.write(message + "\n")
            output_file.write("=" * 80 + "\n")
            return 1


if __name__ == "__main__":
    sys.exit(main())
