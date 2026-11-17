#include <ml/matrix.h>
#include <ml/linreg.h>

Matrix *solve_betahat(const Matrix *X, const Matrix *y)
{
    Matrix *X_tr = mat_transpose(X);
    Matrix *lhs = mat_mul(X_tr, X);
    Matrix *rhs = mat_mul(X_tr, y);
    Matrix *betahat = mat_solve(lhs, rhs);

    mat_free(X_tr);
    mat_free(lhs);
    mat_free(rhs);

    return betahat;
}

Matrix *predict(const Matrix *X, const Matrix *y)
{
    Matrix *betahat = solve_betahat(X, y);
    Matrix *pred = mat_mul(X, betahat);

    mat_free(betahat);

    return pred;
}