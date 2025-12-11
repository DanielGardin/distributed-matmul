// ====================================
// Experiment configs and results
// ====================================
#include <stdlib.h>
#include <stdio.h>

#include <math.h>

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
    char *path;              // path to save a .csv containing the benchmark results
} ExperimentConfig;

typedef struct {
    double *times;
    double *comm_times;
    double *comp_times;
    double overall_time;
    int repeats;
    int correct;
} ExperimentResult;

// Implementation names for user
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

ExperimentResult create_results(const ExperimentConfig *cfg, int rank) {
    double *times = NULL;
    double *comm_times = NULL;
    double *comp_times = NULL;

    if (rank == 0) {
        times = malloc(cfg->repeats * sizeof(double));
        comm_times = calloc(cfg->repeats, sizeof(double));
        comp_times = calloc(cfg->repeats, sizeof(double));
    }

    return (ExperimentResult){
        .times = times,
        .comm_times = comm_times,
        .comp_times = comp_times,
        .repeats = cfg->repeats,
        .correct = 1
    };
}

void log_timings(
    ExperimentResult *result, int iter,
    double total_time, double comm_time, double comp_time, int rank
) {
    if (rank != 0) return;

    if (iter >= result->repeats)
        fprintf(stderr, "WARNING: Trying to log more than %d repeatitions.\n", iter);
    
    result->times[iter] = total_time;
    result->comm_times[iter] = comm_time;
    result->comp_times[iter] = comp_time;
}

double mean(double *array, int size) {
    double mean_val = 0.0;
    for (int i = 0; i < size; i++)
        mean_val += array[i];
    
    return mean_val / size;
}

double std(double *array, int size) {
    double mean_val = mean(array, size);
    double std_val = 0;

    double time_res;
    for (int i=0; i<size; i++) {
        time_res = array[i] - mean_val;
        std_val += time_res * time_res;
    }

    if (std_val > 0) {
        std_val /= (size-1);
        std_val = sqrt(std_val);
    }

    return std_val;
}

double max(double *array, int size) {
    double max_val = array[0];
    double val;

    for (int i = 1; i < size; i++) {
        val = array[i];

        max_val = val > max_val ? val : max_val;
    }

    return max_val;
}

double min(double *array, int size) {
    double min_val = array[0];
    double val;

    for (int i = 1; i < size; i++) {
        val = array[i];

        min_val = val < min_val ? val : min_val;
    }

    return min_val;
}

const char *bars[9] = {
    "_",
    "▁",
    "▂",
    "▃",
    "▄",
    "▅",
    "▆",
    "▇",
    "█"
};
void print_histogram(
    double *array, int size, int n_bins
) {
    double max_val = max(array, size);
    double min_val = min(array, size);

    double width = (max_val - min_val) / n_bins;
    if (width == 0) width = 1;

    int *bin_count = calloc(n_bins, sizeof(int));

    for (int i = 0; i < size; i++) {
        int idx = ((array[i] - min_val) / width);
        if (idx >= n_bins) idx = n_bins - 1;

        bin_count[idx]++;
    }

    int max_count = 0;
    for (int i=0; i<n_bins; i++)
        if (bin_count[i] > max_count) max_count = bin_count[i];

    double freq;
    int height;
    for (int i = 0; i < n_bins; i++) {
        freq = (double)bin_count[i] / max_count;
        height = (int)(8 * freq);
        printf("%s", bars[height]);
    }
    printf("\n");

    free(bin_count);
}

static char *units[4] = {"s ", "ms", "μs", "ns"};
struct Time { double time; char *unit; };
struct Time process_time(double time) {
    int idx = 0;
    while (time < 1 && time > 0 && idx < 4) {
        time *= 1000;
        idx++;
    }

    return (struct Time){.time=time, .unit=units[idx]};
}

void print_results(const ExperimentConfig *cfg, const ExperimentResult *res) {
    struct Time overall_time = process_time(res->overall_time);
    struct Time mean_time = process_time(mean(res->times, res->repeats));

    printf("================     Experiment Results     ================\n");
    printf("%d samples evaluated in a total of %.3lf %s\n", cfg->repeats, overall_time.time, overall_time.unit);
    printf("Matrix size:                        %d x %d\n", cfg->size, cfg->size);
    printf("Block size:                         %d\n", cfg->block_size);
    printf("Warmup rounds:                      %d\n", cfg->warmup_repeats);
    printf("Repetitions:                        %d\n", cfg->repeats);
    if (cfg->use_mpi)
        printf("MPI grid:                           %d x %d\n", cfg->proc_rows, cfg->proc_cols);
    printf("OpenBLAS:                           %s\n", cfg->use_openblas ? "ON" : "OFF");
    printf(
        "Implementation ID:                  %d (%s)\n",
        cfg->implementation_id, impl_name(cfg->implementation_id)
    );
    if (cfg->verify)
        printf("Verification:                       %s%s%s\n",
            res->correct ? "\033[1;32mPASSED" : "\033[1;31mFAILED",
            "\033[0m", "");
    printf("\n");

    struct Time std_time = process_time(std(res->times, res->repeats));
    struct Time max_time = process_time(max(res->times, res->repeats));
    struct Time min_time = process_time(min(res->times, res->repeats));

    printf(
        "Time  (mean ± σ):                   %6.3lf %s ± %6.3lf %s\n",
        mean_time.time, mean_time.unit, std_time.time, std_time.unit
    );
    printf(
        "Time  (min … max):                  %6.3lf %s … %6.3lf %s\n",
        min_time.time, min_time.unit, max_time.time, max_time.unit
    );

    printf("\n      ");
    print_histogram(res->times, res->repeats, 48);
    printf(
        "%3.1lf %s             frequency by time             %3.1lf %s\n", 
        min_time.time, min_time.unit, max_time.time, max_time.unit
    );
    printf("\n");

    
    if (cfg->use_mpi) {
        struct Time mean_comp_time = process_time(mean(res->comp_times, res->repeats));
        struct Time std_comp_time = process_time(std(res->comp_times, res->repeats));
        struct Time mean_comm_time = process_time(mean(res->comm_times, res->repeats));
        struct Time std_comm_time = process_time(std(res->comm_times, res->repeats));

        double *comp_perc = malloc(res->repeats * sizeof(double));
        double *comm_perc = malloc(res->repeats * sizeof(double));

        for (int i = 0; i < res->repeats; i++) {
            comp_perc[i] = res->comp_times[i] / res->times[i];
            comm_perc[i] = res->comm_times[i] / res->times[i];
        }
        double mean_comp_perc = mean(comp_perc, res->repeats);
        double std_comp_perc = std(comp_perc, res->repeats);
        double mean_comm_perc = mean(comm_perc, res->repeats);
        double std_comm_perc = std(comm_perc, res->repeats);

        printf(
            "Comp. time (mean ± σ): %6.2lf %s ± %6.2lf %s  (%3.2lf ± %3.2lf)%%\n",
            mean_comp_time.time, mean_comp_time.unit, std_comp_time.time, std_comp_time.unit,
            mean_comp_perc, std_comp_perc
        );
        printf(
            "Comm. time (mean ± σ): %6.2lf %s ± %6.2lf %s  (%3.2lf ± %3.2lf)%%\n",
            mean_comm_time.time, mean_comm_time.unit, std_comm_time.time, std_comm_time.unit,
            mean_comm_perc, std_comm_perc
        );

        free(comp_perc);
        free(comm_perc);
    }

    printf("============================================================\n");
}

void export_csv(char *path, const ExperimentConfig *cfg, const ExperimentResult *res) {
    FILE *fp = fopen(path, "w");

    fprintf(fp, "rep_id,size,block_size,BLAS,MPI,grid,time");
    if (cfg->use_mpi) {
        fprintf(fp, ", comm_time, comp_time");
    }
    fputc('\n', fp);

    for (int i = 0; i < res->repeats; i++) {
        fprintf(
            fp, "%d, %d, %d, %d, %d, %dx%d, %.9g",
            i, cfg->size, cfg->block_size, cfg->use_openblas, cfg->use_mpi,
            cfg->proc_rows, cfg->proc_cols, res->times[i]
        );
    
        if (cfg->use_mpi) {
            fprintf(
                fp, ", %.9g, %.9g",
                res->comm_times[i], res->comp_times[i]
            );
        }
        fputc('\n', fp);
    }

    fclose(fp);
}
