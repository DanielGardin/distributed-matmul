#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

Matrix create_matrix(int rows, int cols);

void free_matrix(Matrix* matrix);

// static inline float get_element(Matrix *matrix, int row, int col);

// static inline void set_element(Matrix *matrix, int row, int col, float value);

Matrix transpose(const Matrix *matrix);

void print_matrix(const Matrix *matrix);

int is_equal_matrix(const Matrix *A, const Matrix *B, float eps);

#endif // MATRIX_H