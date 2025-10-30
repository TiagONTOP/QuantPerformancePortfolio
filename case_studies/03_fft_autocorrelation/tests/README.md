# Tests Directory

## Structure

- **test_unit.py** - Unit tests validating correctness
- **test_benchmark.py** - Performance benchmarks
- **pytest.ini** - Pytest configuration

## Running Tests

All tests are now compatible with pytest for better test discovery, reporting, and flexibility.

### Prerequisites

Make sure pytest is installed. If using poetry (recommended):

```bash
cd case_studies/03_fft_autocorrelation
poetry install
```

### Unit Tests

```bash
# From the tests directory (with poetry)
cd case_studies/03_fft_autocorrelation/tests
poetry run pytest test_unit.py -v

# Or from the project root
cd case_studies/03_fft_autocorrelation
poetry run pytest tests/test_unit.py -v
```

### Benchmarks

```bash
# From the tests directory (with poetry)
cd case_studies/03_fft_autocorrelation/tests
poetry run pytest test_benchmark.py -v -s

# Or from the project root
cd case_studies/03_fft_autocorrelation
poetry run pytest tests/test_benchmark.py -v -s
```

Note: Use `-s` flag to see the benchmark output details.

### All Tests

```bash
# From the project root
cd case_studies/03_fft_autocorrelation
poetry run pytest tests/ -v
```

### Using pytest directly (if installed globally)

If you have pytest installed in your global Python environment:

```bash
cd case_studies/03_fft_autocorrelation/tests
python -m pytest test_unit.py -v
python -m pytest test_benchmark.py -v -s
```

## Documentation

- See [../TESTS.md](../TESTS.md) for unit test documentation
- See [../BENCHMARKS.md](../BENCHMARKS.md) for benchmark results

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- fft_autocorr (Rust module, compiled with `maturin develop --release`)
