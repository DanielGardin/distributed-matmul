#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

Matrix create_matrix(int rows, int cols) {
    Matrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = calloc(rows * cols, sizeof(float));

    return matrix;
}

void free_matrix(Matrix* matrix) {
    if (matrix) {
        free(matrix->data);
    }
}

// static inline float get_element(Matrix *matrix, int row, int col) {
//     return matrix->data[row * matrix->cols + col];
// }

// static inline void set_element(Matrix *matrix, int row, int col, float value) {
//     matrix->data[row * matrix->cols + col] = value;
// }

Matrix transpose(const Matrix* matrix) {
    Matrix matrix_T = create_matrix(matrix->cols, matrix->rows);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix_T.data[j * matrix_T.cols + i] = matrix->data[i * matrix->cols + j];    
    
    return matrix_T;
}

void print_matrix(const Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i * matrix->cols + j]);
        }
        printf("\n");
    }
}

int is_equal_matrix(const Matrix *A, const Matrix *B, float eps) {
    if (A->cols != B->cols || A->rows != B->rows) return 0;

    int n = A->cols * A->rows;
    for (int i = 0; i < n; i++) {
        float diff = A->data[i] - B->data[i];

        if (diff > eps && diff < -eps)
            return 0;
    }

    return 1;
}