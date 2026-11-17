#include <stdio.h>
#include <stdlib.h>

#include "ml/matrix.h"

Matrix *mat_create(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
        return NULL;

    Matrix *result = malloc(sizeof *result);
    if (!result)
        return NULL;

    result->rows = rows;
    result->cols = cols;

    result->data = malloc(rows * cols * sizeof *result->data);
    if (!result->data)
    {
        free(result);
        return NULL;
    }

    return result;
}

void mat_free(Matrix *m)
{
    if (!m)
        return;
    
    free(m->data);
    m->data = NULL;

    free(m);
    m = NULL;
}

Matrix *mat_add(const Matrix *m1, const Matrix *m2)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols)
        return NULL;

    Matrix *result = mat_create(m1->rows, m1->cols);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows * result->cols; i++)
        result->data[i] = m1->data[i] + m2->data[i];

    return result;
}

Matrix *mat_sub(const Matrix *m1, const Matrix *m2)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols)
        return NULL;

    Matrix *result = mat_create(m1->rows, m1->cols);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows * result->cols; i++)
        result->data[i] = m1->data[i] - m2->data[i];

    return result;
}

Matrix *mat_mul(const Matrix *m1, const Matrix *m2)
{
    if (m1->cols != m2->rows)
        return NULL;

    Matrix *result = mat_create(m1->rows, m2->cols);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows; i++)
    {
        for (size_t j = 0; j < result->cols; j++)
        {
            float dot_result = 0;
            for (size_t x = 0; x < m1->cols; x++)
                dot_result += m1->data[i * m1->cols + x] * m2->data[x * m2->cols + j];
            result->data[i * result->cols + j] = dot_result;
        }
    }

    return result;
}

Matrix *mat_mul_by_const(float mult, const Matrix *m)
{
    Matrix *result = mat_create(m->rows, m->cols);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows * result->cols; i++)
        result->data[i] = mult * m->data[i];

    return result;
}

Matrix *mat_transpose(const Matrix *m)
{
    Matrix *result = mat_create(m->cols, m->rows);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows; i++)
    {
        for (size_t j = 0; j < result->cols; j++)
        {
            result->data[i * result->cols + j] = m->data[j * m->cols + i];
        }
    }

    return result;
}

float mat_trace(const Matrix *m)
{
    size_t min_dim = (m->cols < m->rows) ? m->cols : m->rows;
    float trace = 0;
    for (size_t i = 0; i < min_dim; i++)
        trace += m->data[i * m->cols + i];

    return trace;
}

Matrix *mat_identity(size_t size)
{
    Matrix *result = mat_create(size, size);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->rows; i++)
        for (size_t j = 0; j < result->cols; j++)
            result->data[i * result->cols + j] = (i == j) ? 1 : 0;

    return result;
}