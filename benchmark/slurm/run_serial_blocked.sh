#!/bin/bash
#SBATCH --job-name=matmul_serial_blocked
#SBATCH --time=02:00:00
#SBATCH --output=logs/blocked_run_%a.out
#SBATCH --error=logs/blocked_run_%a.err
#SBATCH --array=0-12

export PATH=/home/daniel.gratti/spack_view/bin:$PATH
export LD_LIBRARY_PATH=/home/daniel.gratti/spack_view/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/daniel.gratti/spack_view/lib:$LIBRARY_PATH

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

BLOCK_SIZE=$((1 << $SLURM_ARRAY_TASK_ID))

start=$(( BLOCK_SIZE > 4 ? BLOCK_SIZE : 4 ))

for ((s = start; s <= 4096; s *= 2)); do
    OUTPUT_FILE="results/blocked/n${s}_b${BLOCK_SIZE}.csv"

    srun ./build/matmul "$OUTPUT_FILE" \
        -n "$s" \
        -b "$BLOCK_SIZE" \
        -i 1 \
        -r 30 \
        -w 5 \
        -s 0
done