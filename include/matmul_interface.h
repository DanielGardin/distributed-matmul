#ifndef MATMUL_INTERFACE_H
#define MATMUL_INTERFACE_H

#include "matrix.h"

int serial_matmul(
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

#endif // MATMUL_INTERFACE_H