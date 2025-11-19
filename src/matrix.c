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
    if (matrix) free(matrix->data);
}

Matrix transpose(const Matrix* matrix) {
    Matrix matrix_T = create_matrix(matrix->cols, matrix->rows);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix_T.data[j * matrix_T.cols + i] = matrix->data[i * matrix->cols + j];    
    
    return matrix_T;
}

void print_matrix(const Matrix* matrix) {
    if (!matrix->data) {
        fprintf(stderr, "ERROR: Cannot print matrix. Data was not initialized.\n");
        return;
    }

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i * matrix->cols + j]);
        }
        printf("\n");
    }
}

int is_equal_matrix(const Matrix *A, const Matrix *B, float rel_tol, float abs_tol) {
    if (A->cols != B->cols || A->rows != B->rows) return 0;

    int n = A->cols * A->rows;
    for (int i = 0; i < n; i++) {
        float a = A->data[i];
        float b = B->data[i];

        float diff = (a > b) ? (a - b) : (b - a);
        float mag_B = (b >= 0) ? b : -b;

        if (diff > abs_tol && diff > rel_tol * mag_B) {
            printf("WARNING: Major divergence between A and B: %f, %f (tolerance: %f)\n",
                   a, b, rel_tol);
            return 0;
        }
    }

    return 1;
}

void random_uniform_matrix(Matrix *mat, float min, float max) {
    float scale = max - min;
    float loc = min;

    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = scale * ((float) rand() / RAND_MAX) + loc;
    }
}

void fill_matrix(Matrix *mat, float value) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = value;
    }
}
