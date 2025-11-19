// Somthing related to unistd.h
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <time.h>

#include <math.h>

#ifdef OPENMPI
#include <mpi.h>
#endif

#define SERIAL_IMPLEMENTATIONS 3

#ifdef OPENMPI
#define NUM_IMPLEMENTATIONS 5
#define CLOCK_FN MPI_Wtime
#else
#define NUM_IMPLEMENTATIONS SERIAL_IMPLEMENTATIONS
#define CLOCK_FN clock
#endif


#include "matrix.h"
#include "matmul_interface.h"

// ====================================
// Experiment configs and results
// ====================================

typedef struct {
    int size;                // global matrix dimension (N)
    int block_size;          // block or tile size
    int warmup_repeats;      // number of non-timed, warmup repetitions
    int repeats;             // number of timed repetitions
    int implementation_id;   // algorithm variant selector
    int seed;                // RNG seed

    int use_openblas;        // use OpenBLAS
    int use_mpi;             // enable MPI distribution

    int proc_rows;           // MPI process grid rows
    int proc_cols;           // MPI process grid cols

    int verify;              // compare result against serial baseline
    char *path;
} ExperimentConfig;

typedef struct {
    double* times;
    int repeats;
    int correct;
} ExperimentResult;

void print_help(const char *program_path);
ExperimentResult run_experiment(const ExperimentConfig* cfg, int rank);
void print_results(const ExperimentConfig *cfg, const ExperimentResult *res);
void export_csv(char *path, const ExperimentConfig *cfg, const ExperimentResult *res);

// ====================================
// Argument parser and main function
// ====================================

int parse_args(int argc, char **argv, ExperimentConfig *cfg) {
    // Positional parameter
    if (argc > 1 && argv[1][0] != '-') {
        cfg->path = argv[1];
        optind = 2;  // skip it so getopt starts from the next argument
    } else {
        cfg->path = NULL;
    }

    int opt;
    while ((opt = getopt(argc, argv, "n:b:w:r:i:s:vh")) != -1) {
        switch (opt) {
            case 'n': cfg->size = atoi(optarg); break;
            case 'b': cfg->block_size = atoi(optarg); break;
            case 'w': cfg->warmup_repeats = atoi(optarg); break;
            case 'r': cfg->repeats = atoi(optarg); break;
            case 's': cfg->seed = atoi(optarg); break;

            case 'i': cfg->implementation_id = atoi(optarg); break;

            case 'v': cfg->verify = 1; break;

            case 'h':
            print_help(argv[0]);
            return 0;

            default:
            fprintf(stderr, "Unknown option '-%c'\n", optopt);
            print_help(argv[0]);
                return 0;
            }
        }

#ifndef OPENMPI
    if (cfg->implementation_id >= SERIAL_IMPLEMENTATIONS) {
        fprintf(
            stderr,
            "ERROR: implementation_id=%d requires MPI support. Please, compile with -DUSE_MPI=1.\n",
            cfg->implementation_id
        );
        return 0;
    }
#endif

    if (cfg->block_size > cfg->size)
        cfg->block_size = cfg->size;

    switch (cfg->implementation_id) {
        case 0: // Serial naive
        case 1: // Serial blocked
            cfg->use_mpi = 0;
            cfg->use_openblas = 0;
            break;

        case 2: // OpenBLAS
            cfg->use_mpi = 0;
            cfg->use_openblas = 1;
            break;

#ifdef OPENMPI
        case 3: // MPI
            cfg->use_mpi = 1;
            cfg->use_openblas = 0;
            break;

        case 4: // OpenBLAS MPI
            cfg->use_mpi = 1;
            cfg->use_openblas = 1;
            break;
#endif

        default:
            fprintf(stderr, "ERROR: Unknown implementation_id=%d.\n", cfg->implementation_id);
            print_help(argv[0]);
            return 0;
    }

    return 1;
}

int main(int argc, char** argv) {
    int rank = 0;

    ExperimentConfig cfg = {
        .size = 512,.block_size = 64, .warmup_repeats = 0,
        .repeats = 3, .implementation_id = 0, .seed = 0,
        .use_openblas = 0, .proc_rows = 1, .proc_cols = 1, .verify = 0
    };

    if (!parse_args(argc, argv, &cfg)) return 1;

// MPI initiallization and grid selection
#ifdef OPENMPI
    MPI_Init(&argc, &argv);
    int n_procs = 1;

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (cfg.use_mpi) {
        int best_r = (int)sqrtf((float)n_procs);
        while (n_procs % best_r) best_r++;

        cfg.proc_rows = best_r;
        cfg.proc_cols = n_procs / best_r;

        if (rank == 0)
            fprintf(
                stderr, "INFO: Selected the best grid %d x %d\n", cfg.proc_rows, cfg.proc_cols
            );
    }
#endif

    ExperimentResult results = run_experiment(&cfg, rank);

    // Only print with the master process
    if (rank == 0) {
        print_results(&cfg, &results);
        
        if (cfg.path) export_csv(cfg.path, &cfg, &results);
    }
    
    free(results.times);

    #ifdef OPENMPI
        MPI_Finalize();
    #endif
    return 0;
}

const char *impl_name(int id) {
    switch (id) {
        case 0: return "Serial naive";
        case 1: return "Serial blocked";
        case 2: return "Serial OpenBLAS";
        case 3: return "MPI SUMMA";
        case 4: return "MPI SUMMA+OpenBLAS";
        default: return "Unknown implementation";
    }
}

void print_help(const char *program_path) {
    fprintf(stderr,
        "Usage: %s [<path>] [options]\n\n"
        "Positional arguments:\n"
        "  <path>               CSV file to save the results (optional)\n\n"
        "Options:\n"
        "  -n <size>            Global matrix dimension (default: 512)\n"
        "  -b <block_size>      Block or tile size (default: 64)\n"
        "  -w <warmup_repeats>  Warm-up repetitions before timing (default: 0)\n"
        "  -r <repeats>         Number of timed repetitions (default: 3)\n"
        "  -i <id>              Implementation ID (see below)\n"
        "  -s <seed>            Random seed (default: 0)\n"
        "  -v                   Verify result against serial baseline\n"
        "  -h                   Show this help message\n\n",
        program_path
    );

    fprintf(stderr, "Implementation IDs:\n");
    for (int i = 0; i < NUM_IMPLEMENTATIONS; i++)
        fprintf(stderr, "  %2d: %s\n", i, impl_name(i));
}

// ====================================
// Algorithm dispatch, run and print
// ====================================

int run_matmul(
    Matrix* A,
    Matrix* B,
    Matrix* C,
    const ExperimentConfig* cfg
) {
    switch (cfg->implementation_id) {
        case 0:
            return naive_matmul(A, B, C);

        case 1:
            return blocked_matmul(A, B, C, cfg->block_size);

        case 2:
            return openblas_matmul(A, B, C);

#ifdef OPENMPI
        case 3:
            return parallel_matmul(
                A, B, C,
                cfg->proc_rows, cfg->proc_cols, cfg->block_size
            );

        case 4:
            return parallel_openblas_matmul(
                A, B, C,
                cfg->proc_rows, cfg->proc_cols, cfg->block_size
            );
#endif
        default:
            return 0;
    }

    return 1;
}

ExperimentResult run_experiment(const ExperimentConfig* cfg, int rank) {
    Matrix A = (Matrix){ .data = NULL, .rows = cfg->size, .cols = cfg->size };
    Matrix B = (Matrix){ .data = NULL, .rows = cfg->size, .cols = cfg->size };
    Matrix C = (Matrix){ .data = NULL, .rows = cfg->size, .cols = cfg->size };

    if (!cfg->use_mpi || rank == 0) {
        A = create_matrix(cfg->size, cfg->size);
        B = create_matrix(cfg->size, cfg->size);
        C = create_matrix(cfg->size, cfg->size);

        srand(cfg->seed);
        random_uniform_matrix(&A, -1, 1);
        random_uniform_matrix(&B, -1, 1);
    }

    // Warm-up runs
    for (int r = 0; r < cfg->warmup_repeats; r ++)
        run_matmul(&A, &B, &C, cfg);

    double *times = NULL;
    if (rank == 0) times = malloc(cfg->repeats * sizeof(double));
    
    for (int r = 0; r < cfg->repeats; r++) {
        if (rank == 0) fill_matrix(&C, 0.0f);    

#ifdef OPENMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        double time = -CLOCK_FN();
        run_matmul(&A, &B, &C, cfg);

#ifdef OPENMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        time += CLOCK_FN();

#ifndef OPENMPI
        time /= CLOCKS_PER_SEC;
#endif
        if (rank == 0) times[r] = time;
    }

    ExperimentResult result = {
        .times = rank==0 ? times : NULL,
        .repeats = cfg->repeats,
        .correct = 1
    };


    if (rank == 0 && cfg->verify) {
        Matrix temp = create_matrix(cfg->size, cfg->size);

        openblas_matmul(&A, &B, &temp);
        result.correct = is_equal_matrix(&C, &temp, 1e-5, 1e-3);

        free_matrix(&temp);
    }

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    return result;
}

void print_results(const ExperimentConfig *cfg, const ExperimentResult *res) {
    printf("===== Experiment Results =====\n");
    printf("Matrix size:        %d x %d\n", cfg->size, cfg->size);
    printf("Block size:         %d\n", cfg->block_size);
    printf("Warmup rounds:      %d\n", cfg->warmup_repeats);
    printf("Repetitions:        %d\n", cfg->repeats);
    if (cfg->use_mpi)
        printf("MPI grid:           %d x %d\n", cfg->proc_rows, cfg->proc_cols);
    printf("OpenBLAS:           %s\n", cfg->use_openblas ? "ON" : "OFF");
    printf("Implementation ID:  %d (%s)\n", cfg->implementation_id, impl_name(cfg->implementation_id));
    printf("\n");

    double mean = 0;
    double std_dev = 0;

    for (int i=0; i<res->repeats; i++) {
        mean += res->times[i];
    }
    mean /= res->repeats;

    if (cfg->repeats > 1) {
        for (int i=0; i<res->repeats; i++) {
            double time_res = res->times[i] - mean;
            std_dev += time_res * time_res;
        }
        std_dev /= (res->repeats - 1);

        std_dev = sqrt(std_dev);
    }

    printf("Mean runtime:       %.6f s\n", mean);
    printf("Std. deviation:     %.6f s\n", std_dev);

    if (cfg->verify)
        printf("Verification:       %s%s%s\n",
            res->correct ? "\033[1;32mPASSED" : "\033[1;31mFAILED",
            "\033[0m", "");

    printf("==============================\n");
}

void export_csv(char *path, const ExperimentConfig *cfg, const ExperimentResult *res) {
    FILE *fp = fopen(path, "w");

    fprintf(fp, "rep_id, size, block_size, BLAS, MPI, grid, time\n");
    for (int i = 0; i < res->repeats; i++) {
        fprintf(
            fp, "%d, %d, %d, %d, %d, %dx%d, %.9g\n",
            i, cfg->size, cfg->block_size, cfg->use_openblas, cfg->use_mpi,
            cfg->proc_rows, cfg->proc_cols, res->times[i]
        );
    }

    fclose(fp);
}
