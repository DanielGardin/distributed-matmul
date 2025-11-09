#include <stdio.h>
#include <cblas.h>

#include "matrix.h"
#include "matmul_interface.h"

int serial_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int block_size
) {
    int n = A->rows;
    int p = A->cols;
    int m = B->cols;

    float* a = A->data;
    float* b = B->data;
    float* c = C->data;

    for (int bi = 0; bi < n; bi+=block_size) {
        for (int bj = 0; bj < m; bj+=block_size) {
            for (int bk=0; bk<p; bk+=block_size){
                int i_max = (bi + block_size < n) ? bi + block_size : n;
                int j_max = (bj + block_size < m) ? bj + block_size : m;
                int k_max = (bk + block_size < p) ? bk + block_size : p;

                for (int i=bi; i < i_max; i++) {
                    for (int k=bk; k < k_max; k++) {
                        for (int j=bj; j < j_max; j++) {
                            c[i * m + j] += a[i * p + k] * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return 1;
}

int serial_matmul_transposed(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int block_size
) {
    int n = A->rows;
    int p = A->cols;
    int m = B->cols;

    Matrix B_T = transpose(B);

    float* a = A->data;
    float* b_t = B_T.data;
    float* c = C->data;

    for (int bi = 0; bi < n; bi+=block_size) {
        for (int bj = 0; bj < m; bj+=block_size) {
            for (int bk=0; bk<p; bk+=block_size){
                int i_max = (bi + block_size < n) ? bi + block_size : n;
                int j_max = (bj + block_size < m) ? bj + block_size : m;
                int k_max = (bk + block_size < p) ? bk + block_size : p;

                for (int i=bi; i < i_max; i++) {
                    for (int j=bj; j < j_max; j++) {
                        for (int k=bk; k < k_max; k++) {
                            c[i * m + j] += a[i * p + k] * b_t[j * p + k];
                        }
                    }
                }
            }
        }
    }

    free_matrix(&B_T);

    return 1;
}


int openblas_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C
) {
    int n = A->rows;
    int p = A->cols;
    int m = B->cols;

    // C = A * B
    cblas_sgemm(
        CblasRowMajor,   // memory layout
        CblasNoTrans,    // A not transposed
        CblasNoTrans,    // B not transposed
        n,               // rows of A
        m,               // cols of B
        p,               // inner dimension
        1.0f,            // alpha
        A->data, p,      // A, lda = cols of A
        B->data, m,      // B, ldb = cols of B
        0.0f,            // beta
        C->data, m       // C, ldc = cols of C
    );

    return 1;
}