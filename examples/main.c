#include <stdio.h>
#include <stdlib.h>

#include "ml/matrix.h"
#include "ml/linreg.h"

int main()
{
    size_t rows = 1000;
    size_t cols = 100;

    float arr1[rows * cols];
    float arr2[cols];

    srand(10);
    for (size_t i = 0; i < rows * cols; i++)
        arr1[i] = (float)rand() / RAND_MAX;

    srand(20);
    for (size_t i = 0; i < rows; i++)
        arr2[i] = (float)rand() / RAND_MAX;

    Matrix X = {rows, cols, arr1};
    Matrix y = {rows, 1, arr2};

    Matrix *pred = predict(&X, &y);

    for (size_t i = 0; i < cols; i++)
        printf("%f ", pred->data[i]);

    printf("\n");

    return 0;
}