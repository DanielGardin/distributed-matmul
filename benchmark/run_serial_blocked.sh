#!/bin/bash
#PBS -N matmul_serial_naive
#PBS -q serial
#PBS -l nodes=1
#PBS -l walltime=01:00:00
#PBS -o logs/blocked_run_^array_index^.out
#PBS -e logs/blocked_run_^array_index^.err
#PBS -J 0-13

cd $PBS_O_WORKDIR

# Build the project if not built yet
if [ ! -f build/matmul ]; then
    ./build.sh
fi

module load gcc/9.4.0
module load openblas/0.3.21-gcc-9.4.0
module load openmpi/4.1.1-gcc-9.4.0

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

BLOCK_SIZE=$((1 << $PBS_ARRAY_INDEX))

start=$(( BLOCK_SIZE > 8 ? BLOCK_SIZE : 8 ))

for ((s = start; s <= 8192; s *= 2)); do
    OUTPUT_FILE="results/raw/blocked/n${s}_b${BLOCK_SIZE}.csv"

    mpirun -np 1 ./build/matmul "$OUTPUT_FILE" \
        -n "$s" \
        -b "$BLOCK_SIZE" \
        -i 1 \
        -r 30 \
        -w 5 \
        -s 0
done
