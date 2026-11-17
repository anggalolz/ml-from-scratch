#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix-inv.h"
#include "ml/matrix.h"

Matrix *mat_solve(const Matrix *m, const Matrix *rhs)
{
    if (m->rows != m->cols)
        return NULL;
    
    Matrix *solution = mat_create(rhs->rows, rhs->cols);
    if (!solution)
        return NULL;
    memcpy(solution->data, rhs->data, rhs->rows * rhs->cols * sizeof *rhs->data);

    Matrix *rref = mat_rref(m, solution);
    if (!rref)
    {
        mat_free(solution);
        return NULL;
    }

    for (size_t i = 0; i < rref->cols; i++)
    {
        if (fabs(rref->data[i * rref->cols + i]) < TOL)
        {
            mat_free(rref);
            mat_free(solution);
            return NULL;
        }
    }

    mat_free(rref);
    return solution;
}

Matrix *mat_inv(const Matrix *m)
{
    if (m->rows != m->cols)
        return NULL;

    Matrix *identity = mat_identity(m->rows);
    if (!identity)
        return NULL;

    Matrix *inv = mat_solve(m, identity);

    mat_free(identity);
    return inv;
}

Matrix *mat_row_echelon(const Matrix *m, const Matrix *rhs)
{
    if (rhs && m->rows != rhs->rows)
        return NULL;

    Matrix *result = mat_create(m->rows, m->cols);
    if (!result)
        return NULL;
    memcpy(result->data, m->data, result->rows * result->cols * sizeof *result->data);

    size_t h = 0;
    size_t k = 0;

    while (h < result->rows && k < result->cols)
    {
        size_t i_max = _max_abs_pivot_row_index(result, k, h);
        if (fabs(result->data[i_max * result->cols + k]) < TOL)
        {
            k++;
            continue;
        }

        _swap_rows(result, h, i_max);
        if (rhs)
            _swap_rows(rhs, h, i_max);
        
        for (size_t i = h + 1; i < result->rows; i++)
        {
            float mult = -result->data[i * result->cols + k] / result->data[h * result->cols + k];
            if (fabs(mult) < TOL)
                continue;
            
            _add_a_row_multiple(result, i, h, mult);
            if (rhs)
                _add_a_row_multiple(rhs, i, h, mult);
        }

        h++;
        k++;
    }

    return result;
}

Matrix *mat_rref(const Matrix *m, const Matrix *rhs)
{
    size_t h = 0;
    size_t k = 0;

    Matrix *result = mat_row_echelon(m, rhs);
    if (!result)
        return NULL;
    
    while (h < result->rows && k < result->cols)
    {
        float pivot = result->data[h * result->cols + k];
        if (fabs(pivot) < TOL)
        {
            k++;
            continue;
        }

        _multiply_row_by_const(result, h, 1 / pivot);
        if (rhs)
            _multiply_row_by_const(rhs, h, 1 / pivot);
        
        for (size_t i = 0; i < h; i++)
        {
            float mult = -result->data[i * result->cols + k];
            if (fabs(mult) < TOL)
                continue;
            
            _add_a_row_multiple(result, i, h, mult);
            if (rhs)
                _add_a_row_multiple(rhs, i, h, mult);
        }

        h++;
        k++;
    }

    return result;
}

int mat_det(const Matrix *m, float *result)
{
    if (m->rows != m->cols)
        return -1;

    Matrix *ref = mat_row_echelon(m, NULL);
    if (!ref)
        return -1;
    
    *result = 1;
    for (size_t i = 0; i < ref->cols; i++)
        *result *= ref->data[i * ref->cols + i];
    
    mat_free(ref);

    return 0;
}

void _swap_rows(const Matrix *m, size_t index1, size_t index2)
{
    float temp;
    for (size_t j = 0; j < m->cols; j++)
    {
        temp = m->data[index1 * m->cols + j];
        m->data[index1 * m->cols + j] = m->data[index2 * m->cols + j];
        m->data[index2 * m->cols + j] = temp;
    }
}

void _add_a_row_multiple(const Matrix *m, size_t index1, size_t index2, float mult)
{
    for (size_t j = 0; j < m->cols; j++)
        m->data[index1 * m->cols + j] += mult * m->data[index2 * m->cols + j];
}

void _multiply_row_by_const(const Matrix *m, size_t index, float mult)
{
    for (size_t j = 0; j < m->cols; j++)
        m->data[index * m->cols + j] *= mult;
}

size_t _max_abs_pivot_row_index(const Matrix *m, size_t col_index, size_t start)
{
    size_t max_index = start;
    for (size_t i = start; i < m->rows; i++)
        if (fabs(m->data[i * m->cols + col_index]) > fabs(m->data[max_index * m->cols + col_index]))
            max_index = i;

    return max_index;
}