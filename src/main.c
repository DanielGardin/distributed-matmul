#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>

// #include <unistd.h>
#include <getopt.h>
#include <time.h>


#ifdef OPENMPI
#include <mpi.h>
#endif

#include "matrix.h"
#include "matmul_interface.h"

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
} ExperimentConfig;

typedef struct {
    double* times;
    int repeats;
    int correct;
} ExperimentResult;


int parse_args(int argc, char **argv, ExperimentConfig *cfg);

ExperimentResult run_experiment(const ExperimentConfig* cfg);

void print_results(const ExperimentConfig *cfg, const ExperimentResult *res);

int main(int argc, char** argv) {
    ExperimentConfig cfg = {0};

    if (!parse_args(argc, argv, &cfg)) return 1;

#ifdef OPENMPI
    MPI_Init(&argc, &argv);
#endif

    // select between MATMUL_SERIAL or MATMUL_MPI via arguments
    ExperimentResult results = run_experiment(&cfg);

#ifdef OPENMPI
    MPI_Finalize();
#endif

    print_results(&cfg, &results);
    free(results.times);
    return 0;
}

void print_help(char *program_path) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  -n <size>           Global matrix dimension (default: 512)\n"
        "  -b <block_size>     Block or tile size (default: 64)\n"
        "  -w <warmup_repeats> Warm-up repetitions before timing (default: 0)\n"
        "  -r <repeats>        Number of timed repetitions (default: 3)\n"
        "  -i <id>             Implementation ID (see below)\n"
        "  -s <seed>           Random seed (default: 0)\n"
        "  -p <proc_rows>      MPI process grid rows (default: 1)\n"
        "  -c <proc_cols>      MPI process grid cols (default: 1)\n"
        "  -v                  Verify result against serial baseline\n"
        "  -h                  Show this help message\n\n"
        "Implementation IDs:\n"
        "  0 : Serial blocked multiplication\n"
        "  1 : Serial transposed multiplication\n"
        "  2 : OpenBLAS (cblas_sgemm) baseline\n",
        program_path);
}

int parse_args(int argc, char **argv, ExperimentConfig *cfg) {
    // Default values
    cfg->size              = 512;
    cfg->block_size        = 64;
    cfg->warmup_repeats    = 0;
    cfg->repeats           = 3;
    cfg->implementation_id = 0;
    cfg->seed              = 0;

    cfg->use_openblas      = 0;
    cfg->proc_rows         = 1;
    cfg->proc_cols         = 1;

    cfg->verify            = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:b:w:r:i:S:p:c:vh")) != -1) {
        switch (opt) {
            case 'n': cfg->size = atoi(optarg); break;
            case 'b': cfg->block_size = atoi(optarg); break;
            case 'w': cfg->warmup_repeats = atoi(optarg); break;
            case 'r': cfg->repeats = atoi(optarg); break;
            case 's': cfg->seed = atoi(optarg); break;

            case 'i': cfg->implementation_id = atoi(optarg); break;

            case 'p': cfg->proc_rows = atoi(optarg); break;
            case 'c': cfg->proc_cols = atoi(optarg); break;

            case 'v': cfg->verify = 1; break;

            case 'h':
            default:
                print_help(argv[0]);
                return 0;
        }
    }

    switch (cfg->implementation_id) {
        case 0: // Serial blocked
        case 1: // Serial blocked transposed
            cfg->use_mpi = 0;
            cfg->use_openblas = 0;
            break;
        
        case 2: // OpenBLAS
            cfg->use_mpi = 0;
            cfg->use_openblas = 1;
            break;
        
        default:
            fprintf(stderr, "Error: Unknown implementation_id=%d.\n", cfg->implementation_id);
            print_help(argv[0]);
            return 0;
    }

    return 1;
}

void initialize_matrix(Matrix *mat) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = 2 * ((float) rand() / RAND_MAX) - 1;
    }
}

void zero_matrix(Matrix *mat) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = 0;
    }
}

int dispatch_implementation(
    Matrix* A,
    Matrix* B,
    Matrix* C,
    const ExperimentConfig* cfg
) {
    if (!cfg->use_mpi) {
        if (cfg->use_openblas) return openblas_matmul(A, B, C);
        else return serial_matmul(A, B, C, cfg->block_size);
    }

    return 1;
}

ExperimentResult run_experiment(const ExperimentConfig* cfg) {
    Matrix A = create_matrix(cfg->size, cfg->size);
    Matrix B = create_matrix(cfg->size, cfg->size);
    Matrix C = create_matrix(cfg->size, cfg->size);

    srand(cfg->seed);
    initialize_matrix(&A);
    initialize_matrix(&B);

    for (int r = 0; r < cfg->warmup_repeats; r ++)
        dispatch_implementation(&A, &B, &C, cfg);

    double *times = malloc(cfg->repeats * sizeof(double));
    for (int r = 0; r < cfg->repeats; r++) {
        zero_matrix(&C);

        clock_t start = clock();
        dispatch_implementation(&A, &B, &C, cfg);
        clock_t end = clock();
        times[r] = (double)(end - start) / CLOCKS_PER_SEC;
    }

    int correct = 1;
    if (cfg->verify) {
        Matrix temp = create_matrix(cfg->size, cfg->size);

        openblas_matmul(&A, &B, &temp);
        correct = is_equal_matrix(&C, &temp, 1e-6);

        free_matrix(&temp);
    }

    ExperimentResult result = {
        .times = times,
        .repeats = cfg->repeats,
        .correct = correct
    };

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    return result;
}

const char *impl_name(int id) {
    switch (id) {
        case 0: return "Serial blocked";
        case 1: return "Serial transposed";
        case 2: return "OpenBLAS";
        case 3: return "Parallel block (MPI)";
        default: return "Unknown implementation";
    }
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
    }


    printf("Mean runtime:       %.6f s\n", mean);
    printf("Std. deviation:     %.6f s\n", std_dev);
    // printf("GFLOPs:             %.3f\n", res->gflops);

    if (cfg->verify)
        printf("Verification:       %s%s%s\n",
            res->correct ? "\033[1;32mPASSED" : "\033[1;31mFAILED",
            "\033[0m", "");
    printf("==============================\n");
}