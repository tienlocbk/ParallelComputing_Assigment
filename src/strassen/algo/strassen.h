#pragma once

#include "../../utils/matrix.h"

enum { STRASSEN_DEFAULT_CUTOFF = 512, STRASSEN_PRODUCTS = 7 };

/*
 * strassen.h — Strassen matrix multiplication
 *
 * Public API:
 *   Matrix *strassen_multiply(const Matrix *A, const Matrix *B, int cutoff);
 *
 * - A and B must be square matrices of the same size.
 * - Internally pads to the next power of 2 if needed.
 * - Falls back to naive multiply when size <= cutoff (default 512).
 */

/* Public entry point (handles padding / unpadding). */
#ifdef __cplusplus
extern "C" {
#endif
Matrix *strassen_multiply(const Matrix *A, const Matrix *B, int cutoff);

/* A leaf writes the full product into output and returns nonzero on success. */
typedef int (*StrassenLeaf)(const Matrix *a, const Matrix *b, Matrix *output,
                            void *context);
Matrix *strassen_multiply_with_leaf(const Matrix *a, const Matrix *b, int cutoff,
                                    StrassenLeaf leaf, void *context);

/* Shared with the OpenMP implementation. Product indices run from 0 to 6. */
int strassen_size(const Matrix *a, const Matrix *b);
int strassen_padded_size(int size);
void strassen_form_operands(const Matrix *a, const Matrix *b, int index,
                            Matrix *left, Matrix *right);
void strassen_accumulate(Matrix *output, const Matrix *product, int index);

#ifdef __cplusplus
}
#endif
