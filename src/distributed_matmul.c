#ifdef OPENMPI
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <cblas.h>

#include "matrix.h"
#include "matmul_interface.h"

#define ceildiv(a, b) (((a) + (b) - 1) / (b))
#define min(a, b) ((a < b ? a : b))


// Global variables to store communication and computation times
double g_comm = 0.0;
double g_comp = 0.0;


int get_block_size(int dim, int n_procs, int proc_idx) {
    int base_size = dim / n_procs;
    int remainder = dim % n_procs;
    if (proc_idx < remainder) base_size++;
    return base_size;
}

int get_block_start(int dim, int n_procs, int proc_idx) {
    int base_size = dim / n_procs;
    int remainder = dim % n_procs;
    return base_size * proc_idx + (proc_idx < remainder ? proc_idx : remainder);
}

static void subarraycopy(
    const float *source, int start_row, int start_col,
    int rows, int cols, int orig_cols, int output_cols, float* output
) {
    const float *src = &source[start_row * orig_cols + start_col];
    float *dst = output;

    for (int i = 0; i < rows; i++) {
        memcpy(dst, src, cols * sizeof(float));
        src += orig_cols;
        dst += output_cols;
    }
}

void send_submatrices(
    const Matrix* restrict A,
    const Matrix* restrict B,
    int n_proc_rows, int n_proc_cols,
    int rank,
    Matrix *localA, Matrix *localB
) {
    int N = A->rows;
    int P = A->cols;
    int M = B->cols;

    if (rank == 0) {
        float *temp_A = malloc((N/n_proc_rows + 1) * (P/n_proc_cols + 1) * sizeof(float));
        float *temp_B = malloc((P/n_proc_rows + 1) * (M/n_proc_cols + 1) * sizeof(float));

        for (int pr = 0; pr < n_proc_rows; pr++) {
            for (int pc = 0; pc < n_proc_cols; pc++) {
                int dest = pr * n_proc_cols + pc;

                int dest_N = get_block_size(N, n_proc_rows, pr);
                int dest_PA = get_block_size(P, n_proc_cols, pc);
                int row_start_A = get_block_start(N, n_proc_rows, pr);
                int col_start_A = get_block_start(P, n_proc_cols, pc);

                int dest_PB = get_block_size(P, n_proc_rows, pr);
                int dest_M = get_block_size(M, n_proc_cols, pc);
                int row_start_B = get_block_start(P, n_proc_rows, pr);
                int col_start_B = get_block_start(M, n_proc_cols, pc);

                if (dest == 0) {
                    subarraycopy(A->data, row_start_A, col_start_A, dest_N, dest_PA, P, dest_PA, localA->data);
                    subarraycopy(B->data, row_start_B, col_start_B, dest_PB, dest_M, M, dest_M, localB->data);

                } else {
                    subarraycopy(A->data, row_start_A, col_start_A, dest_N, dest_PA, P, dest_PA, temp_A);
                    MPI_Send(temp_A, dest_N * dest_PA, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);

                    subarraycopy(B->data, row_start_B, col_start_B, dest_PB, dest_M, M, dest_M, temp_B);
                    MPI_Send(temp_B, dest_PB * dest_M, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
                }
            }
        }

        free(temp_A);
        free(temp_B);

    } else {
        MPI_Recv(localA->data, localA->rows * localA->cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localB->data, localB->rows * localB->cols, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
}

void recv_submatrices(
    const Matrix *local, int n_proc_rows, int n_proc_cols, int rank, const Matrix *result
) {
    if (rank == 0) {
        int N = result->rows;
        int M = result->cols;

        float *temp = malloc((N/n_proc_rows + 1) * (M/n_proc_cols + 1) * sizeof(float));

        for (int pr = 0; pr < n_proc_rows; pr++) {
            for (int pc = 0; pc < n_proc_cols; pc++) {
                int dest = pr * n_proc_cols + pc;

                int dest_N = get_block_size(N, n_proc_rows, pr);
                int dest_M = get_block_size(M, n_proc_cols, pc);

                int start_i = get_block_start(N, n_proc_rows, pr);
                int start_j = get_block_start(M, n_proc_cols, pc);

                float *result_ptr = result->data + start_i * result->cols + start_j;
                if (dest == 0) {
                    subarraycopy(local->data, 0, 0, local->rows, local->cols, local->cols, result->cols, result_ptr);

                } else {
                    MPI_Recv(temp, dest_N * dest_M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    subarraycopy(temp, 0, 0, dest_N, dest_M, dest_M, result->cols, result_ptr);
                }
            }
        }

        free(temp);

    } else {
        MPI_Send(local->data, local->rows * local->cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}

// Implements SUMMA
int parallel_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int n_proc_rows,
    int n_proc_cols,
    int block_size
) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int proc_row = rank / n_proc_cols;
    int proc_col = rank % n_proc_cols;

    int N = A->rows;
    int P = A->cols;
    int M = B->cols;

    int local_N = get_block_size(N, n_proc_rows, proc_row);
    int local_PA = get_block_size(P, n_proc_cols, proc_col);
    int local_PB = get_block_size(P, n_proc_rows, proc_row);
    int local_M = get_block_size(M, n_proc_cols, proc_col);

    Matrix block_A = create_matrix(local_N, local_PA);
    Matrix block_B = create_matrix(local_PB, local_M);
    Matrix block_C = create_matrix(local_N, local_M);

    double T0 = MPI_Wtime();
    send_submatrices(A, B, n_proc_rows, n_proc_cols, rank, &block_A, &block_B);
    g_comm += MPI_Wtime() - T0;


    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, proc_row, proc_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, proc_col, proc_row, &col_comm);

    int k = 0;

    int ownerA = 0;
    int startA = 0;
    int endA = get_block_size(P, n_proc_cols, 0);

    int ownerB = 0;
    int startB = 0;
    int endB = get_block_size(P, n_proc_rows, 0);

    float *panelA = malloc(local_N * block_size * sizeof(float));
    float *panelB = malloc(block_size * local_M * sizeof(float));

    while (k < P) {
        int nbA = endA - k;
        int nbB = endB - k;

        int panel_size = min(min(block_size, nbA), nbB);

        if (proc_col == ownerA)
        subarraycopy(block_A.data, 0, k - startA, block_A.rows, panel_size, block_A.cols, panel_size, panelA);

        if (proc_row == ownerB)
        subarraycopy(block_B.data, k - startB, 0, panel_size, block_B.cols, block_B.cols, block_B.cols, panelB);

        T0 = MPI_Wtime();
        MPI_Bcast(panelA, local_N * panel_size, MPI_FLOAT, ownerA, row_comm);
        MPI_Bcast(panelB, panel_size * local_M, MPI_FLOAT, ownerB, col_comm);
        g_comm += MPI_Wtime() - T0;

        T0 = MPI_Wtime();
        naive_matmul(
            &(Matrix){local_N, panel_size, panelA},
            &(Matrix){panel_size, local_M, panelB},
            &block_C
        );
        g_comp += MPI_Wtime() - T0;

        k += panel_size;

        while (k >= endA) {
            ownerA++;
            startA = endA;
            endA += get_block_size(P, n_proc_cols, ownerA);
        }

        while (k >= endB) {
            ownerB++;
            startB = endB;
            endB += get_block_size(P, n_proc_rows, ownerB);
        }
    }

    free(panelA);
    free(panelB);
    free_matrix(&block_A);
    free_matrix(&block_B);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    T0 = MPI_Wtime();
    recv_submatrices(&block_C, n_proc_rows, n_proc_cols, rank, C);
    g_comm += MPI_Wtime() - T0;

    free_matrix(&block_C);

    return 1;
}

int parallel_openblas_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int n_proc_rows,
    int n_proc_cols,
    int block_size
) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int proc_row = rank / n_proc_cols;
    int proc_col = rank % n_proc_cols;

    int N = A->rows;
    int P = A->cols;
    int M = B->cols;

    int local_N = get_block_size(N, n_proc_rows, proc_row);
    int local_PA = get_block_size(P, n_proc_cols, proc_col);
    int local_PB = get_block_size(P, n_proc_rows, proc_row);
    int local_M = get_block_size(M, n_proc_cols, proc_col);

    Matrix block_A = create_matrix(local_N, local_PA);
    Matrix block_B = create_matrix(local_PB, local_M);
    Matrix block_C = create_matrix(local_N, local_M);

    double T0 = MPI_Wtime();
    send_submatrices(A, B, n_proc_rows, n_proc_cols, rank, &block_A, &block_B);
    g_comm += MPI_Wtime() - T0;


    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, proc_row, proc_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, proc_col, proc_row, &col_comm);

    int k = 0;

    int ownerA = 0;
    int startA = 0;
    int endA = get_block_size(P, n_proc_cols, 0);

    int ownerB = 0;
    int startB = 0;
    int endB = get_block_size(P, n_proc_rows, 0);

    float *panelA = malloc(local_N * block_size * sizeof(float));
    float *panelB = malloc(block_size * local_M * sizeof(float));

    while (k < P) {
        int nbA = endA - k;
        int nbB = endB - k;
        
        int panel_size = min(min(block_size, nbA), nbB);

        if (proc_col == ownerA)
        subarraycopy(block_A.data, 0, k - startA, block_A.rows, panel_size, block_A.cols, panel_size, panelA);
        
        if (proc_row == ownerB)
        subarraycopy(block_B.data, k - startB, 0, panel_size, block_B.cols, block_B.cols, block_B.cols, panelB);
        
        T0 = MPI_Wtime();
        MPI_Bcast(panelA, local_N * panel_size, MPI_FLOAT, ownerA, row_comm);
        MPI_Bcast(panelB, panel_size * local_M, MPI_FLOAT, ownerB, col_comm);
        g_comm += MPI_Wtime() - T0;

        T0 = MPI_Wtime();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            local_N, local_M, panel_size,
            1.0f,
            panelA, panel_size,
            panelB, local_M,
            1.0f,
            block_C.data, local_M
        );
        g_comp += MPI_Wtime() - T0;

        k += panel_size;

        while (k >= endA) {
            ownerA++;
            startA = endA;
            endA += get_block_size(P, n_proc_cols, ownerA);
        }

        while (k >= endB) {
            ownerB++;
            startB = endB;
            endB += get_block_size(P, n_proc_rows, ownerB);
        }
    }

    free(panelA);
    free(panelB);
    free_matrix(&block_A);
    free_matrix(&block_B);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    T0 = MPI_Wtime();
    recv_submatrices(&block_C, n_proc_rows, n_proc_cols, rank, C);
    g_comm += MPI_Wtime() - T0;

    free_matrix(&block_C);

    return 1;
}

#endif
