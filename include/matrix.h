#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

Matrix create_matrix(int rows, int cols);

void free_matrix(Matrix* matrix);

Matrix transpose(const Matrix *matrix);

void print_matrix(const Matrix *matrix);

int is_equal_matrix(const Matrix *A, const Matrix *B, float rel_tol, float abs_tol);

void random_uniform_matrix(Matrix *mat, float min, float max);

void fill_matrix(Matrix *mat, float value);

#endif // MATRIX_H
