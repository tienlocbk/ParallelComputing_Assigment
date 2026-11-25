#pragma once

#include "../algo/strassen.h"

/* Multiplies equally sized square matrices using OpenMP tasks.
 * The cutoff has the same meaning and default as in strassen_multiply.
 * The caller owns the returned matrix. Returns NULL on failure.
 */
#ifdef __cplusplus
extern "C"
#endif
Matrix *strassen_multiply_openmp(const Matrix *A, const Matrix *B, int cutoff);
