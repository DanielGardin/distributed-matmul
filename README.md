# Distributed Matrix Multiplication

A high-performance C implementation of distributed matrix multiplication with multiple algorithmic approaches.

## Overview

This project implements and benchmarks various matrix multiplication algorithms, from naive implementations to optimized distributed approaches using MPI. The implementations are designed for dense square matrix multiplication and include both serial and parallel variants.

## Features

- **Multiple Implementation Strategies:**
  - Naive serial multiplication
  - Blocked/tiled serial multiplication (cache-optimized)
  - OpenBLAS-based serial multiplication
  - SUMMA (Scalable Universal Matrix Multiplication Algorithm) - distributed
  - SUMMA with OpenBLAS blocks - optimized distributed

- **Performance Analysis:**
  - Detailed benchmarking framework with warmup and timing phases
  - CSV-based result export for post-analysis
  - Communication vs. computation time tracking
  - Configurable experimental parameters

- **Flexibility:**
  - Optional MPI support (compile with or without distributed computing)
  - Configurable matrix sizes and block sizes
  - SUMMA implementation for an arbitrary number of processes
  - Result verification against serial baseline
  - Adjustable number of repetitions and warmup iterations

## Building

### Prerequisites

- GCC/Clang C11 compiler
- CMake 3.12+
- OpenBLAS library
- OpenMPI (optional, for distributed execution)

### Build Instructions

**Serial version only:**
```bash
mkdir build
cd build
cmake -B . -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**With MPI support:**
```bash
./build.sh
```

Or manually:
```bash
mkdir build
cd build
cmake -B . -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=1
cmake --build .
```

The compiled executable will be at `build/matmul`.

## Usage

### Basic Syntax

```bash
./build/matmul [<csv_file>] [options]
```

### Options

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `-n` | `<size>` | 512 | Global matrix dimension (rows and columns) |
| `-b` | `<block_size>` | 64 | Block/tile size for algorithms |
| `-w` | `<warmup_repeats>` | 0 | Warm-up iterations before timing |
| `-r` | `<repeats>` | 3 | Number of timed repetitions |
| `-i` | `<id>` | - | Implementation ID (see below) |
| `-s` | `<seed>` | 0 | Random seed for matrix initialization |
| `-v` | - | - | Verify result against serial baseline |
| `-h` | - | - | Show help message |

### Implementation IDs

- `0`: Naive serial (O(n³))
- `1`: Blocked serial (cache-optimized)
- `2`: OpenBLAS serial
- `3`: SUMMA (requires MPI)
- `4`: SUMMA with OpenBLAS (requires MPI)

### Examples

Run naive 512×512 multiplication with verification:
```bash
./build/matmul -n 512 -i 0 -v
```

Run SUMMA algorithm with 2048×2048 matrix, 256-block size, 2 warmups, 5 repetitions, save to results.csv:
```bash
mpirun -np 4 ./build/matmul results.csv -n 2048 -b 256 -w 2 -r 5 -i 3
```

## Project Structure

```
.
├── CMakeLists.txt              # Build configuration
├── build.sh                    # Build script for MPI version
├── src/
│   ├── main.c                  # Argument parser and entry point
│   ├── matrix.c/h              # Matrix utilities (allocation, initialization)
│   ├── serial_matmul.c         # Naive and blocked serial implementations
│   ├── distributed_matmul.c    # SUMMA implementation
│   └── experiment.c            # Benchmarking and timing framework
├── include/                    # Public headers
├── benchmark/                  # Benchmark scripts
├── results/                    # Raw results directory
│   └── raw/
│       ├── naive/              # Results from naive implementation
│       ├── blocked/            # Results from blocked implementation
│       ├── openblas/           # Results from OpenBLAS
│       ├── summa/              # Results from SUMMA
│       └── summablas/          # Results from SUMMA+OpenBLAS
└── build/                      # Build artifacts (generated)
```

## Benchmarking

Predefined benchmark scripts are available in the `benchmark/` directory:

- `run_serial_naive.sh` - Benchmark naive serial algorithm
- `run_serial_blocked.sh` - Benchmark blocked serial algorithm
- `run_serial_openblas.sh` - Benchmark OpenBLAS implementation
- `run_summa_rankone.sh` - Benchmark SUMMA algorithm
- `run_summa_oblas_rankone.sh` - Benchmark SUMMA with OpenBLAS

Results are saved as CSV files in `results/raw/` organized by implementation type.

## Performance Notes

- **Naive serial:** Slow, memory inefficient. Use only for correctness verification.
- **Blocked serial:** 2-3x faster than naive on large matrices due to cache locality.
- **OpenBLAS:** Highly optimized BLAS implementation, fastest serial method.
- **SUMMA:** Distributed algorithm. Performance depends on matrix size, block size, and network bandwidth.
- **SUMMA+OpenBLAS:** Combines distributed algorithm with optimized local computation.

Block size selection is critical for performance. Typical values: 64-512 depending on available memory and cache size.

## Development Notes

- The code uses POSIX C11 standard
- Compilation flags: `-O3 -march=native` for release builds
- MPI is conditionally compiled via `OPENMPI` preprocessor definition
- Timing uses `MPI_Wtime()` when MPI is available, `clock()` otherwise
