#ifndef MATRIX_INV_H
#define MATRIX_INV_H

#include <stdio.h>

#include "ml/matrix.h"

void _swap_rows(const Matrix *m, size_t index1, size_t index2);

void _add_a_row_multiple(const Matrix *m, size_t index1, size_t index2, float mult);

void _multiply_row_by_const(const Matrix *m, size_t index, float mult);

size_t _max_abs_pivot_row_index(const Matrix *m, size_t col_index, size_t start);

#endif