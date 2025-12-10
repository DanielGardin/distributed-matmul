#!/bin/bash
#PBS -N matmul_summa_rank_one
#PBS -q paralela
#PBS -l walltime=72:00:00
#PBS -l nodes=2:ppn=128
#PBS -o logs/summa_rone_run.out
#PBS -e logs/summa_rone_run.err

cd $PBS_O_WORKDIR

# Build the project if not built yet
if [ ! -f build/matmul ]; then
    ./build.sh
fi

module purge
module load gcc/9.4.0
module load openblas/0.3.21-gcc-9.4.0
module load openmpi/4.1.1-gcc-9.4.0

export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

MAX_PROCS=256
for ((i=2; i<=16; i++)); do
    NPROCS=$((i * i))
    N_PARALLEL_JOBS=$((MAX_PROCS / NPROCS))
    (( N_PARALLEL_JOBS < 1 )) && N_PARALLEL_JOBS=1

    s=8
    while (( s <= 4096 )); do
        for ((k=0; k<N_PARALLEL_JOBS && s<=8192; k++)); do
            OUTPUT_FILE="results/raw/summa/np${NPROCS}_n${s}_b1.csv"
            mpirun --map-by socket --bind-to core -np "$NPROCS" \
                ./build/matmul "$OUTPUT_FILE" \
                -n "$s" -b 1 -i 3 -r 30 -w 5 -s 0 &
            s=$((s * 2))
        done
        wait
    done
done










for ((i=2; i<=16; i++)); do

    NPROCS=$((i * i))
    N_PARALLEL_JOBS=$((MAX_PROCS / NPROCS))
    (( N_PARALLEL_JOBS < 1 )) && N_PARALLEL_JOBS=1

    STEP=$((1 << (N_PARALLEL_JOBS - 1)))

    for ((s0=8; s0<=4096; s0*=STEP)); do

        s=$s0
        for ((k=0; k<N_PARALLEL_JOBS && s<=8192; k++)); do

            OUTPUT_FILE="results/raw/summa/np${NPROCS}_n${s}_b1.csv"

            mpirun --map-by socket --bind-to core -np "$NPROCS" \
                ./build/matmul "$OUTPUT_FILE" \
                -n "$s" -b 1 -i 3 -r 30 -w 5 -s 0 &

            s=$((s * 2))
        done

        wait
    done
done
