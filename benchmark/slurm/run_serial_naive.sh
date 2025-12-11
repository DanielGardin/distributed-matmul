#!/bin/bash
#SBATCH --job-name=matmul_serial_naive
#SBATCH --time=02:00:00
#SBATCH --output=logs/naive_run_%a.out
#SBATCH --error=logs/naive_run_%a.err
#SBATCH --array=0-10

export PATH=/home/daniel.gratti/spack_view/bin:$PATH
export LD_LIBRARY_PATH=/home/daniel.gratti/spack_view/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/daniel.gratti/spack_view/lib:$LIBRARY_PATH

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

SIZES=(4 8 16 32 64 128 256 512 1024 2048 4096)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_FILE=results/naive/n${SIZE}.csv

srun ./build/matmul "$OUTPUT_FILE" \
    -n "$SIZE" \
    -i 0 \
    -r 30 \
    -w 5 \
    -s 0
