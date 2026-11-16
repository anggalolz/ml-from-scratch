#ifndef LINREG_H
#define LINREG_H

#include <ml/matrix.h>

Matrix *solve_betahat(const Matrix *X, const Matrix *y);

Matrix *predict(const Matrix *X, const Matrix *y);

#endif