#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

#define TOL 1e-10

typedef struct Matrix
{
    size_t rows, cols;
    float *data;
} Matrix;

// Memory management

Matrix *mat_create(size_t rows, size_t cols);

void mat_free(Matrix *m);

// Basic matrices operation

Matrix *mat_add(const Matrix *m1, const Matrix *m2);

Matrix *mat_sub(const Matrix *m1, const Matrix *m2);

Matrix *mat_mul(const Matrix *m1, const Matrix *m2);

Matrix *mat_mul_by_const(float mult, const Matrix *m);

Matrix *mat_transpose(const Matrix *m);

float mat_trace(const Matrix *m);

Matrix *mat_identity(size_t size);

// Inverse-related operation

Matrix *mat_solve(const Matrix *m, const Matrix *rhs);

Matrix *mat_inv(const Matrix *m);

Matrix *mat_row_echelon(const Matrix *m, const Matrix *rhs);

Matrix *mat_rref(const Matrix *m, const Matrix *rhs);

int mat_det(const Matrix *m, float *out);

#endif