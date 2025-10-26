"""
Benchmark script to determine optimal FFT setting for ACF computation.
Run this to update the processing.py file with the optimal setting.
"""

import sys
from pathlib import Path

# Add suboptimal directory to path
sys.path.insert(0, str(Path(__file__).parent / "suboptimal"))

from processing import benchmark_acf_methods, determine_optimal_fft_setting


def main():
    print("=" * 70)
    print("ACF FFT BENCHMARK - COMPREHENSIVE TEST")
    print("=" * 70)
    print()

    # Run comprehensive benchmark
    results = benchmark_acf_methods(
        series_sizes=[100, 500, 1000, 5000, 10000, 50000, 100000],
        max_lag=50,
        n_iterations=100
    )

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    # Analyze results
    print("\nSpeedup Analysis (fft=True vs fft=False):")
    for size in sorted(results["fft_false"].keys()):
        time_false = results["fft_false"][size]
        time_true = results["fft_true"][size]
        speedup = time_false / time_true
        faster = "FFT" if speedup > 1 else "Direct"
        print(f"  Size {size:>7}: {speedup:>6.2f}x faster with {faster}")

    # Determine optimal setting
    print("\n" + "=" * 70)
    print("OPTIMAL SETTING DETERMINATION")
    print("=" * 70)

    use_fft, reasoning = determine_optimal_fft_setting(threshold_size=1000)

    print("\nRECOMMENDATION:")
    print(f"  Use fft={use_fft} for optimal performance")

    # Update the processing.py file
    print("\n" + "=" * 70)
    print("UPDATING processing.py")
    print("=" * 70)

    processing_file = Path(__file__).parent / "suboptimal" / "processing.py"
    content = processing_file.read_text()

    # Replace the fft parameter in compute_autocorrelation
    old_line = "autocorr_values = acf(series, nlags=max_lag, fft=False)"
    new_line = f"autocorr_values = acf(series, nlags=max_lag, fft={use_fft})"

    if old_line in content or "fft=True" in content:
        # Replace either fft=False or fft=True
        content = content.replace("fft=False)", f"fft={use_fft})")
        content = content.replace("fft=True)", f"fft={use_fft})")
        processing_file.write_text(content)
        print(f"✓ Updated compute_autocorrelation to use fft={use_fft}")
    else:
        print("✗ Could not find the line to update")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
