# Tests Directory

## Structure

- **test_unit.py** - Unit tests validating correctness
- **test_benchmark.py** - Performance benchmarks

## Running Tests

### Unit Tests

```bash
python tests/test_unit.py
```

### Benchmarks

```bash
python tests/test_benchmark.py
```

### All Tests

```bash
python tests/test_unit.py && python tests/test_benchmark.py
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
