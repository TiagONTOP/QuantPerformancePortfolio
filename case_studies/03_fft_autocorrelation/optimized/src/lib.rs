use num_complex::Complex64;
use numpy::{PyArray1, PyReadonlyArray1};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::collections::HashMap;
use std::sync::Mutex;

// -------- FFT Plan Cache --------
struct Plan {
    r2c: std::sync::Arc<dyn RealToComplex<f64>>,
    c2r: std::sync::Arc<dyn ComplexToReal<f64>>,
    scratch_fwd: Vec<Complex64>,
    scratch_inv: Vec<Complex64>,
}

static PLAN_CACHE: OnceCell<Mutex<HashMap<usize, Plan>>> = OnceCell::new();

fn get_plan(m: usize) -> Plan {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Check if plan exists
    if let Some(p) = cache.lock().unwrap().get(&m) {
        return Plan {
            r2c: p.r2c.clone(),
            c2r: p.c2r.clone(),
            scratch_fwd: vec![Complex64::new(0.0, 0.0); p.r2c.get_scratch_len()],
            scratch_inv: vec![Complex64::new(0.0, 0.0); p.c2r.get_scratch_len()],
        };
    }

    // Create new plan
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(m);
    let c2r = planner.plan_fft_inverse(m);

    let plan = Plan {
        r2c: r2c.clone(),
        c2r: c2r.clone(),
        scratch_fwd: vec![Complex64::new(0.0, 0.0); r2c.get_scratch_len()],
        scratch_inv: vec![Complex64::new(0.0, 0.0); c2r.get_scratch_len()],
    };

    // Store in cache (without scratch buffers to save memory)
    cache.lock().unwrap().insert(
        m,
        Plan {
            r2c,
            c2r,
            scratch_fwd: vec![],
            scratch_inv: vec![],
        },
    );

    plan
}

// -------- Fast Length (2,3,5,7-smooth) --------
fn is_smooth_2357(mut n: usize) -> bool {
    for p in [2, 3, 5, 7] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

fn next_fast_len(mut n: usize) -> usize {
    while !is_smooth_2357(n) {
        n += 1;
    }
    n
}

// -------- Direct Autocorrelation (for small max_lag) --------
/// Compute autocorrelation directly using O(n*max_lag) algorithm.
/// This is faster than FFT when max_lag is small relative to n.
fn autocorr_direct_norm(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();

    // Compute mean and center data
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut xx = Vec::with_capacity(n);
    xx.extend(x.iter().map(|&v| v - mean));

    // Compute lag-0 variance
    let var0 = xx.iter().map(|&v| v * v).sum::<f64>() / n as f64;

    let mut out = Vec::with_capacity(max_lag);
    for k in 1..=max_lag {
        let mut s = 0.0f64;
        let limit = n - k;

        // Tight loop for correlation computation
        for i in 0..limit {
            s += xx[i] * xx[i + k];
        }

        let c = s / n as f64;
        out.push(if var0 != 0.0 { c / var0 } else { 0.0 });
    }

    out
}

// -------- FFT Autocorrelation (R2C, in-place, cached) --------
/// Compute autocorrelation using Real FFT with optimized memory usage.
/// Uses R2C/C2R transforms and in-place operations to minimize allocations.
fn autocorr_fft_norm(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;

    // Size for linear correlation using 2357-smooth length
    let m = next_fast_len(2 * n - 1);

    // Time buffer: length m, zero-padded
    let mut time = vec![0.0f64; m];

    // Center data in place
    time[..n]
        .iter_mut()
        .zip(x.iter())
        .for_each(|(t, &v)| *t = v - mean);

    // Get cached plan and scratch buffers
    let mut plan = get_plan(m);

    // Forward R2C transform
    let mut freq = plan.r2c.make_output_vec();
    plan.r2c
        .process_with_scratch(&mut time, &mut freq, &mut plan.scratch_fwd)
        .unwrap();

    // Compute power spectrum in place: |X|^2
    for z in &mut freq {
        let re = z.re;
        let im = z.im;
        *z = Complex64::new(re * re + im * im, 0.0);
    }

    // Inverse C2R transform
    let mut time_back = plan.c2r.make_output_vec();
    plan.c2r
        .process_with_scratch(&mut freq, &mut time_back, &mut plan.scratch_inv)
        .unwrap();

    // Normalize by m (IFFT scaling)
    let scale = 1.0 / (m as f64);
    for t in &mut time_back {
        *t *= scale;
    }

    // Extract normalized autocorrelation
    let lag0 = time_back[0];
    let denom = if lag0.abs() > 1e-18 { lag0 } else { 1.0 };

    let end = (max_lag + 1).min(n);
    let mut out = Vec::with_capacity(end - 1);
    for k in 1..end {
        out.push(time_back[k] / denom);
    }

    out
}

// -------- Adaptive Strategy: Direct vs FFT --------
/// Automatically choose between direct and FFT methods based on problem size.
/// Direct method is faster for small max_lag, FFT is faster for large max_lag.
fn autocorr_adaptive(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let m = next_fast_len(2 * n - 1);

    // Heuristic threshold: when to use direct vs FFT
    // FFT cost ~ m*log(m), Direct cost ~ n*max_lag
    // Calibrated factor of 0.6 based on empirical testing
    let k_threshold = ((m as f64).log2() * (m as f64) / (n as f64) * 0.6).ceil() as usize;

    if max_lag <= k_threshold {
        autocorr_direct_norm(x, max_lag)
    } else {
        autocorr_fft_norm(x, max_lag)
    }
}

/// Compute autocorrelation for a time series using optimized FFT or direct method.
///
/// This function automatically selects the best algorithm (direct O(n*k) or FFT O(n log n))
/// based on the input size and max_lag. It uses Real FFT with 2357-smooth lengths,
/// in-place operations, and cached FFT plans for optimal performance.
///
/// The implementation matches scipy.signal.correlate behavior while providing
/// superior performance for most use cases, especially with small to medium max_lag.
///
/// Parameters
/// ----------
/// series : ndarray
///     Input time series as a 1D numpy array of float64
/// max_lag : int, optional
///     Maximum lag to compute autocorrelation for (default=1)
///
/// Returns
/// -------
/// ndarray
///     Array of autocorrelation values from lag 1 to max_lag
///
/// Examples
/// --------
/// >>> import fft_autocorr
/// >>> import numpy as np
/// >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
/// >>> result = fft_autocorr.compute_autocorrelation(data, max_lag=3)
/// >>> print(result)
/// [0.7  0.41212121  0.14848485]
///
/// Performance Notes
/// -----------------
/// - Uses direct method for small max_lag (typically < 100)
/// - Uses Real FFT (R2C/C2R) with 2357-smooth lengths for larger max_lag
/// - Caches FFT plans for repeated calls with similar sizes
/// - Releases Python GIL during computation for better concurrency
#[pyfunction]
#[pyo3(signature = (series, max_lag=1))]
fn compute_autocorrelation<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Validate inputs
    if max_lag == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_lag must be >= 1",
        ));
    }

    let x = series.as_slice()?;
    if x.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input array cannot be empty",
        ));
    }

    // Release GIL and compute (allows concurrent Python threads to run)
    let result = py.allow_threads(|| autocorr_adaptive(x, max_lag));

    Ok(PyArray1::from_vec_bound(py, result))
}

/// FFT-based autocorrelation computation module.
///
/// This module provides high-performance autocorrelation computation using Rust's FFT
/// implementation, exposed to Python via PyO3.
#[pymodule]
fn fft_autocorr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_autocorrelation, m)?)?;
    Ok(())
}
