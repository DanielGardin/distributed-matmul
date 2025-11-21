#!/bin/bash
#PBS -N matmul_serial_openblas
#PBS -q serial
#PBS -l nodes=1
#PBS -l walltime=02:00:00
#PBS -o logs/openblas_run_^array_index^.out
#PBS -e logs/openblas_run_^array_index^.err
#PBS -J 0-11

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

SIZES=(4 8 16 32 64 128 256 512 1024 2048 4096 8192)

SIZE=${SIZES[$PBS_ARRAY_INDEX]}

OUTPUT_FILE="results/raw/openblas/n${SIZE}.csv"

mpirun -np 1 ./build/matmul "$OUTPUT_FILE" \
    -n "$SIZE" \
    -i 0 \
    -r 30 \
    -w 5 \
    -s 0
