#!/bin/bash
#SBATCH --job-name=matmul_summa
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/summa_36_%a.out
#SBATCH --error=logs/summa_36_%a.err
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
    OUTPUT_FILE="results/summa/np36_n${s}_b${BLOCK_SIZE}.csv"

    mpirun -np 36 ./build/matmul "$OUTPUT_FILE" \
        -n "$s" \
        -b "$BLOCK_SIZE" \
        -i 3 \
        -r 30 \
        -w 5 \
        -s 0
done
