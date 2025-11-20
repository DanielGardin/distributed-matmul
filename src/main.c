// ====================================
// Argument parser and main function
// ====================================

// Somthing related to unistd.h
#define _POSIX_C_SOURCE 200809L

#include <unistd.h>
#include <time.h>

#ifdef OPENMPI
#include <mpi.h>
#endif

#define SERIAL_IMPLEMENTATIONS 3
#define PARALLEL_IMPLEMENTATIONS 2

#ifdef OPENMPI
#define CLOCK_FN MPI_Wtime
#define COMPILE_T 1
#define BARRIER(comm) MPI_Barrier(comm)
#else
#define CLOCK_FN clock
#define COMPILE_T 0
#define BARRIER(comm) ((void)0)
double g_comm = 0.0;
double g_comp = 0.0;
#endif

#include "matrix.h"
#include "matmul_interface.h"

#include "experiment.c"

ExperimentResult run_experiment(const ExperimentConfig *cfg, int rank);

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
    for (int i = 0; i < SERIAL_IMPLEMENTATIONS + COMPILE_T * PARALLEL_IMPLEMENTATIONS; i++)
        fprintf(stderr, "  %2d: %s\n", i, impl_name(i));
}

int parse_args(int argc, char **argv, ExperimentConfig *cfg, int rank) {
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
            if (rank == 0) {
                fprintf(
                    stderr,
                    (
                        cfg->implementation_id > 0 &&
                        cfg->implementation_id < SERIAL_IMPLEMENTATIONS + PARALLEL_IMPLEMENTATIONS
                    ) ?
                    "ERROR: implementation_id=%d requires MPI support. Please, compile with -DUSE_MPI=1.\n" :
                    "ERROR: Unknown implementation_id=%d.\n",
                    cfg->implementation_id
                );
                print_help(argv[0]);
            }
            return 0;
    }

#ifdef OPENMPI
    int n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    if (cfg->use_mpi) {
        int best_r = (int)ceil(sqrtf((float)n_procs));
        while (n_procs % best_r) best_r++;

        cfg->proc_rows = best_r;
        cfg->proc_cols = n_procs / best_r;

        if (rank == 0)
            fprintf(
                stderr, "INFO: Selected the best grid %d x %d\n", cfg->proc_rows, cfg->proc_cols
            );

    } else if (n_procs > 1 && rank == 0) {
        fprintf(
            stderr, "INFO: Running more than one process in a serial algorithm. "
            "The results in every rank, besides the root, will be ignored.\n"
        );
    }
#endif

    return 1;
}

int main(int argc, char** argv) {
    double T0 = CLOCK_FN();
    int rank = 0;

#ifdef OPENMPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    ExperimentConfig cfg = {
        .size = 512,.block_size = 64, .warmup_repeats = 0,
        .repeats = 3, .implementation_id = 0, .seed = 0,
        .use_openblas = 0, .proc_rows = 1, .proc_cols = 1, .verify = 0
    };

    if (!parse_args(argc, argv, &cfg, rank)) return 1;

    ExperimentResult results = run_experiment(&cfg, rank);

    results.overall_time = (double)CLOCK_FN() - T0;

#ifndef OPENMPI
    results.overall_time /= CLOCKS_PER_SEC;
#endif

    // Only print with the master process
    if (rank == 0) {
        print_results(&cfg, &results);

        if (cfg.path) export_csv(cfg.path, &cfg, &results);
    }
    
    free(results.times);
    free(results.comm_times);
    free(results.comp_times);

    #ifdef OPENMPI
        MPI_Finalize();
    #endif
    return 0;
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

inline void warmup_runs(Matrix *A, Matrix *B, Matrix *C, const ExperimentConfig *cfg) {
    for (int r = 0; r < cfg->warmup_repeats; r ++)
        run_matmul(A, B, C, cfg);
}

ExperimentResult run_experiment(const ExperimentConfig *cfg, int rank) {
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

    warmup_runs(&A, &B, &C, cfg);

    ExperimentResult results = create_results(cfg, rank);
    
    double comm_time = 0.0;
    double comp_time = 0.0;
    for (int r = 0; r < cfg->repeats; r++) {
        if (rank == 0) {
            fill_matrix(&C, 0.0f);
            random_uniform_matrix(&A, -1, 1);
            random_uniform_matrix(&B, -1, 1);
        }
        g_comm = 0.0;
        g_comp = 0.0;

        BARRIER(MPI_COMM_WORLD);

        double time = -CLOCK_FN();
        run_matmul(&A, &B, &C, cfg);

        BARRIER(MPI_COMM_WORLD);

        time += CLOCK_FN();

#ifdef OPENMPI
        MPI_Reduce(&g_comm, &comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&g_comp, &comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
        time /= CLOCKS_PER_SEC;
#endif

        if (rank == 0 && cfg->verify) {
            Matrix temp = create_matrix(cfg->size, cfg->size);

            openblas_matmul(&A, &B, &temp);
            results.correct = results.correct && is_equal_matrix(&C, &temp, 1e-5, 1e-3);

            free_matrix(&temp);
        }

        log_timings(&results, r, time, comm_time, comp_time, rank);
    }



    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    return results;
}
