#!/bin/bash
#SBATCH --job-name=matmul_summa_optimal
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/summa_naive_optimal_%a.out
#SBATCH --error=logs/summa_naive_optimal_%a.err
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
    OUTPUT_FILE="results/summablas/np16_n${s}_b${BLOCK_SIZE}.csv"

    mpirun -np 16 ./build/matmul "$OUTPUT_FILE" \
        -n "$s" \
        -b 1 \
        -i 3 \
        -r 30 \
        -w 5 \
        -s 0
done
