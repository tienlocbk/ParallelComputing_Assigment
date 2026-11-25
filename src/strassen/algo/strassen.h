#pragma once

#include "../../utils/matrix.h"

/*
 * strassen.h — Strassen matrix multiplication
 *
 * Public API:
 *   Matrix *strassen_multiply(const Matrix *A, const Matrix *B, int cutoff);
 *
 * - A and B must be square matrices of the same size.
 * - Internally pads to the next power of 2 if needed.
 * - Falls back to naive multiply when size <= cutoff (default 64).
 */

/* Public entry point (handles padding / unpadding). */
#ifdef __cplusplus
extern "C"
#endif
Matrix *strassen_multiply(const Matrix *A, const Matrix *B, int cutoff);
