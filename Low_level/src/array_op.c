#include "array_operations.h"
#include <omp.h>
#include <immintrin.h> // For SIMD instructions

#define TILE_SIZE 16 // Define a suitable tile size for loop tiling

void matrix_multiply(double *a, double *b, double *result, int a_rows, int a_cols, int b_cols)
{
    // Initialize result matrix
    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < b_cols; j++)
        {
            result[i * b_cols + j] = 0.0;
        }
    }

// Matrix multiplication with loop tiling and parallelization
#pragma omp parallel for collapse(2)
    for (int i = 0; i < a_rows; i += TILE_SIZE)
    {
        for (int j = 0; j < b_cols; j += TILE_SIZE)
        {
            for (int k = 0; k < a_cols; k += TILE_SIZE)
            {
                for (int ii = i; ii < i + TILE_SIZE && ii < a_rows; ii++)
                {
                    for (int jj = j; jj < j + TILE_SIZE && jj < b_cols; jj++)
                    {
                        double sum = 0.0;
                        for (int kk = k; kk < k + TILE_SIZE && kk < a_cols; kk++)
                        {
                            sum += a[ii * a_cols + kk] * b[kk * b_cols + jj];
                        }
                        result[ii * b_cols + jj] += sum;
                    }
                }
            }
        }
    }
}
