use num_complex::Complex64;
use numpy::{PyArray1, PyReadonlyArray1};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;

// -------- Thread-local Buffer Pool --------
// Reusable buffers per thread to avoid allocations
struct BufferSet {
    time: Vec<f64>,
    freq: Vec<Complex64>,
    time_back: Vec<f64>,
    scratch_fwd: Vec<Complex64>,
    scratch_inv: Vec<Complex64>,
}

thread_local! {
    static BUFFER_POOL: RefCell<HashMap<usize, BufferSet>> = RefCell::new(HashMap::new());
}

// --- MODIFIED : Changed from `get_or_create_buffers` to `with_buffers` loan pattern ---
/// Borrows a set of reusable buffers from a thread-local pool for a given size `m`.
///
/// This function takes a closure `f` and executes it, passing a mutable reference
/// to a `BufferSet`. This avoids heap allocations on every call by reusing
/// the `Vec`s' capacity.
fn with_buffers<F, R>(
    m: usize,
    r2c_scratch_len: usize,
    c2r_scratch_len: usize,
    mut f: F,
) -> R
where
    F: FnMut(&mut BufferSet) -> R,
{
    BUFFER_POOL.with(|pool| {
        let mut map = pool.borrow_mut();

        // Get or create the buffer set for this size
        let bufset = map.entry(m).or_insert_with(|| BufferSet {
            time: Vec::new(),
            freq: Vec::new(),
            time_back: Vec::new(),
            scratch_fwd: Vec::new(),
            scratch_inv: Vec::new(),
        });

        // Resize buffers to correct size (reusing existing capacity)
        // .resize() is efficient if capacity is already sufficient
        bufset.time.clear();
        bufset.time.resize(m, 0.0);

        bufset.freq.clear();
        bufset.freq.resize(m / 2 + 1, Complex64::new(0.0, 0.0));

        bufset.time_back.clear();
        bufset.time_back.resize(m, 0.0);

        bufset.scratch_fwd.clear();
        bufset
            .scratch_fwd
            .resize(r2c_scratch_len, Complex64::new(0.0, 0.0));

        bufset.scratch_inv.clear();
        bufset
            .scratch_inv
            .resize(c2r_scratch_len, Complex64::new(0.0, 0.0));

        // Execute the closure, "lending" it the buffers
        f(bufset)
    })
    // bufset (and map/pool borrows) go out of scope, but the Vecs
    // remain in the thread-local HashMap with their capacity intact.
}
// --- END MODIFIED ---

// -------- FFT Plan Cache --------
// (No changes to this section, it is correct)
struct Plan {
    r2c: std::sync::Arc<dyn RealToComplex<f64>>,
    c2r: std::sync::Arc<dyn ComplexToReal<f64>>,
}

static PLAN_CACHE: OnceCell<Mutex<HashMap<usize, Plan>>> = OnceCell::new();

fn get_plan(m: usize) -> Plan {
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Check if plan exists
    if let Some(p) = cache.lock().unwrap().get(&m) {
        return Plan {
            r2c: p.r2c.clone(),
            c2r: p.c2r.clone(),
        };
    }

    // Create new plan
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(m);
    let c2r = planner.plan_fft_inverse(m);

    let plan = Plan {
        r2c: r2c.clone(),
        c2r: c2r.clone(),
    };

    // Store in cache
    cache.lock().unwrap().insert(m, plan.clone());

    plan
}

impl Clone for Plan {
    fn clone(&self) -> Self {
        Plan {
            r2c: self.r2c.clone(),
            c2r: self.c2r.clone(),
        }
    }
}

// -------- Fast Length (2,3,5,7-smooth) --------
// (No changes to this section, it is correct)
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
// (No changes to this section, it is correct)
fn autocorr_direct_norm(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();

    // Single-pass: compute mean and center data simultaneously
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut xx = Vec::with_capacity(n);
    let mut var0 = 0.0f64;

    for &v in x {
        let centered = v - mean;
        var0 += centered * centered;
        xx.push(centered);
    }
    var0 /= n as f64;

    // Check for constant series (zero variance)
    let is_constant = var0.abs() < 1e-14;

    // Compute available lags (limited by series length)
    let available_lags = (n - 1).min(max_lag);

    // Parallel computation across lags
    let use_parallel = (available_lags as u64) * (n as u64) > 100_000;

    let mut result = if use_parallel && available_lags > 10 {
        (1..=available_lags)
            .into_par_iter()
            .map(|k| {
                if is_constant {
                    f64::NAN
                } else {
                    let limit = n - k;
                    let mut s = 0.0f64;
                    let mut i = 0;

                    // Unrolled loop (4-way)
                    while i + 4 <= limit {
                        s += xx[i] * xx[i + k]
                            + xx[i + 1] * xx[i + k + 1]
                            + xx[i + 2] * xx[i + k + 2]
                            + xx[i + 3] * xx[i + k + 3];
                        i += 4;
                    }

                    // Remainder
                    while i < limit {
                        s += xx[i] * xx[i + k];
                        i += 1;
                    }

                    let c = s / n as f64;
                    c / var0
                }
            })
            .collect()
    } else {
        // Sequential version
        let mut out = Vec::with_capacity(available_lags);
        for k in 1..=available_lags {
            if is_constant {
                out.push(f64::NAN);
            } else {
                let limit = n - k;
                let mut s = 0.0f64;
                let mut i = 0;

                // Unrolled loop (4-way)
                while i + 4 <= limit {
                    s += xx[i] * xx[i + k]
                        + xx[i + 1] * xx[i + k + 1]
                        + xx[i + 2] * xx[i + k + 2]
                        + xx[i + 3] * xx[i + k + 3];
                    i += 4;
                }

                // Remainder
                while i < limit {
                    s += xx[i] * xx[i + k];
                    i += 1;
                }

                let c = s / n as f64;
                out.push(c / var0);
            }
        }
        out
    };

    // Pad with NaN
    while result.len() < max_lag {
        result.push(f64::NAN);
    }

    result
}

// -------- FFT Autocorrelation (R2C, in-place, cached) --------
// --- MODIFIED : Adapted to use the `with_buffers` closure pattern ---
fn autocorr_fft_norm(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();

    // Single-pass: compute mean and check for constant series
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    // Check for constant series (zero variance)
    if variance.abs() < 1e-14 {
        return vec![f64::NAN; max_lag];
    }

    // Size for linear correlation
    let m = next_fast_len(2 * n - 1);

    // Get cached plan
    let plan = get_plan(m);

    // "Borrow" thread-local buffers and execute FFT logic inside the closure
    let out = with_buffers(
        m,
        plan.r2c.get_scratch_len(),
        plan.c2r.get_scratch_len(),
        |buffers| {
            // `buffers` is a &mut BufferSet, reusing pooled memory

            // Center the data and zero-pad
            for i in 0..n {
                buffers.time[i] = x[i] - mean;
            }
            // Zero-pad the rest (already done by .resize(m, 0.0) in with_buffers)
            // for i in n..m {
            //     buffers.time[i] = 0.0;
            // }

            // Forward R2C transform
            plan.r2c
                .process_with_scratch(
                    &mut buffers.time,
                    &mut buffers.freq,
                    &mut buffers.scratch_fwd,
                )
                .unwrap();

            // Compute power spectrum in place: |X|^2
            let freq_len = buffers.freq.len();
            if freq_len > 1000 {
                buffers.freq.par_iter_mut().for_each(|z| {
                    let magnitude_sq = z.re * z.re + z.im * z.im;
                    *z = Complex64::new(magnitude_sq, 0.0);
                });
            } else {
                for z in &mut buffers.freq {
                    let magnitude_sq = z.re * z.re + z.im * z.im;
                    *z = Complex64::new(magnitude_sq, 0.0);
                }
            }

            // Inverse C2R transform
            plan.c2r
                .process_with_scratch(
                    &mut buffers.freq,
                    &mut buffers.time_back,
                    &mut buffers.scratch_inv,
                )
                .unwrap();

            // Normalize by m (IFFT scaling)
            let scale = 1.0 / (m as f64);
            if m > 10000 {
                buffers.time_back.par_iter_mut().for_each(|t| *t *= scale);
            } else {
                for t in &mut buffers.time_back {
                    *t *= scale;
                }
            }

            // Extract normalized autocorrelation
            let lag0 = buffers.time_back[0];

            let available_lags = (n - 1).min(max_lag);
            // This 'out' vec is the only new allocation needed for the result
            let mut out = Vec::with_capacity(max_lag); 

            for k in 1..=available_lags {
                out.push(buffers.time_back[k] / lag0);
            }

            // Pad with NaN
            while out.len() < max_lag {
                out.push(f64::NAN);
            }

            out // Return the result vector from the closure
        },
    );

    out // Return the final result
}
// --- END MODIFIED ---

// -------- Adaptive Strategy: Direct vs FFT --------
// (No changes to this section, it is correct)
fn autocorr_adaptive(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let m = next_fast_len(2 * n - 1);

    // Heuristic threshold
    let fft_cost_estimate = (m as f64) * (m as f64).log2();
    let direct_cost_estimate = (n as f64) * (max_lag as f64) * 0.25; // 4-way unroll factor
    let fft_total_cost = fft_cost_estimate + 1000.0; // Fixed overhead

    if direct_cost_estimate * 1.2 < fft_total_cost {
        autocorr_direct_norm(x, max_lag)
    } else {
        autocorr_fft_norm(x, max_lag)
    }
}

// -------- PyO3 Bindings --------
// (No changes to this section, it is correct)
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

    // Release GIL and compute
    let result = py.allow_threads(|| autocorr_adaptive(x, max_lag));

    Ok(PyArray1::from_vec_bound(py, result))
}

#[pymodule]
fn fft_autocorr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_autocorrelation, m)?)?;
    Ok(())
}