#include <cblas.h>
#include "matmul_interface.h"

int naive_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C
) {
    int n = A->rows;
    int p = A->cols;
    int m = B->cols;

    float *a = A->data;
    float *b = B->data;
    float *c = C->data;

    for (int i=0; i < n; i++) {
        for (int k=0; k < p; k++) {
            for (int j=0; j < m; j++) {
                c[i * m + j] += a[i * p + k] * b[k * m + j];
            }
        }
    }

    return 1;
}

int blocked_matmul(
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
                            c[i * m + j] += a[i * p + k] * b[k * m + j];
                        }
                    }
                }
            }
        }
    }

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

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        n, m, p,
        1.0f,
        A->data, p,
        B->data, m,
        0.0f,
        C->data, m
    );

    return 1;
}
