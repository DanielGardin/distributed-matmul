#ifndef MATMUL_INTERFACE_H
#define MATMUL_INTERFACE_H

#include "matrix.h"

int naive_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C
);

int blocked_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int block_size
);

int openblas_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C
);

extern double g_comm;
extern double g_comp;

#ifdef OPENMPI
int parallel_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int n_proc_rows,
    int n_proc_cols,
    int block_size
);

int parallel_openblas_matmul(
    const Matrix* restrict A,
    const Matrix* restrict B,
    Matrix* C,
    int n_proc_rows,
    int n_proc_cols,
    int block_size
);

#endif

#endif // MATMUL_INTERFACE_H
