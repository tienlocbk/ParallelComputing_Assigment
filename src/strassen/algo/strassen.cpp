#include "strassen.h"
#include <string.h>

/*
 * strassen.cpp — Strassen matrix multiplication (sequential)
 *
 * Algorithm overview (for n×n matrices, n = power of 2):
 *
 *   Partition A and B into four n/2 × n/2 sub-matrices each:
 *
 *       A = | A11  A12 |      B = | B11  B12 |
 *           | A21  A22 |          | B21  B22 |
 *
 *   Compute 7 products :
 *
 *       M1 = (A11 + A22) × (B11 + B22)
 *       M2 = (A21 + A22) × B11
 *       M3 = A11 × (B12 - B22)
 *       M4 = A22 × (B21 - B11)
 *       M5 = (A11 + A12) × B22
 *       M6 = (A21 - A11) × (B11 + B12)
 *       M7 = (A12 - A22) × (B21 + B22)
 *
 *   Combine into result:
 *
 *       C11 = M1 + M4 - M5 + M7
 *       C12 = M3 + M5
 *       C21 = M2 + M4
 *       C22 = M1 - M2 + M3 + M6
 *
 */

/* ------------------------------------------------------------------ */
/*  Helper: extract sub-matrix (copy)                                  */
/*  Copies the half-size block at (row_off, col_off) from src.         */
/* ------------------------------------------------------------------ */
static Matrix *get_submatrix(const Matrix *src, int row_off, int col_off, int size)
{
    Matrix *sub = mat_create(size, size);
    if (!sub) return NULL;

    for (int i = 0; i < size; i++) {
        memcpy(&MAT_AT(sub, i, 0),
               &MAT_AT(src, row_off + i, col_off),
               (size_t)size * sizeof(double));
    }
    return sub;
}

/* ------------------------------------------------------------------ */
/*  Helper: write sub-matrix back into a larger matrix                 */
/* ------------------------------------------------------------------ */
static void set_submatrix(Matrix *dst, int row_off, int col_off,
                          const Matrix *sub, int size)
{
    for (int i = 0; i < size; i++) {
        memcpy(&MAT_AT(dst, row_off + i, col_off),
               &MAT_AT(sub, i, 0),
               (size_t)size * sizeof(double));
    }
}

/* ------------------------------------------------------------------ */
/*  Strassen                                                          */
/*  Precondition: A and B are n×n, n is a power of 2.                  */
/* ------------------------------------------------------------------ */
static Matrix *strassen(const Matrix *A, const Matrix *B, int n, int cutoff)
{
    /* Base case: to naive multiply */
    if (n <= cutoff) {
        return (*A) * (*B);
    }

    int half = n / 2;

    /* ---- Split A and B into quadrants ---- */
    Matrix *A11 = get_submatrix(A, 0,    0,    half);
    Matrix *A12 = get_submatrix(A, 0,    half, half);
    Matrix *A21 = get_submatrix(A, half, 0,    half);
    Matrix *A22 = get_submatrix(A, half, half, half);

    Matrix *B11 = get_submatrix(B, 0,    0,    half);
    Matrix *B12 = get_submatrix(B, 0,    half, half);
    Matrix *B21 = get_submatrix(B, half, 0,    half);
    Matrix *B22 = get_submatrix(B, half, half, half);

    /* ---- Compute the 7 Strassen products ---- */

    /* M1 = (A11 + A22) × (B11 + B22) */
    Matrix *T1 = *A11 + *A22;
    Matrix *T2 = *B11 + *B22;
    Matrix *M1 = strassen(T1, T2, half, cutoff);
    mat_free(T1); mat_free(T2);

    /* M2 = (A21 + A22) × B11 */
    T1 = *A21 + *A22;
    Matrix *M2 = strassen(T1, B11, half, cutoff);
    mat_free(T1);

    /* M3 = A11 × (B12 - B22) */
    T1 = mat_sub(B12, B22);
    Matrix *M3 = strassen(A11, T1, half, cutoff);
    mat_free(T1);

    /* M4 = A22 × (B21 - B11) */
    T1 = mat_sub(B21, B11);
    Matrix *M4 = strassen(A22, T1, half, cutoff);
    mat_free(T1);

    /* M5 = (A11 + A12) × B22 */
    T1 = *A11 + *A12;
    Matrix *M5 = strassen(T1, B22, half, cutoff);
    mat_free(T1);

    /* M6 = (A21 - A11) × (B11 + B12) */
    T1 = mat_sub(A21, A11);
    T2 = *B11 + *B12;
    Matrix *M6 = strassen(T1, T2, half, cutoff);
    mat_free(T1); mat_free(T2);

    /* M7 = (A12 - A22) × (B21 + B22) */
    T1 = mat_sub(A12, A22);
    T2 = *B21 + *B22;
    Matrix *M7 = strassen(T1, T2, half, cutoff);
    mat_free(T1); mat_free(T2);

    /* Free sub-matrices of A and B — no longer needed */
    mat_free(A11); mat_free(A12); mat_free(A21); mat_free(A22);
    mat_free(B11); mat_free(B12); mat_free(B21); mat_free(B22);

    /* ---- Combine into C quadrants ---- */

    /* C11 = M1 + M4 - M5 + M7 */
    Matrix *C11 = *M1 + *M4;
    mat_sub_inplace(C11, M5);
    mat_add_inplace(C11, M7);

    /* C12 = M3 + M5 */
    Matrix *C12 = *M3 + *M5;

    /* C21 = M2 + M4 */
    Matrix *C21 = *M2 + *M4;

    /* C22 = M1 - M2 + M3 + M6 */
    Matrix *C22 = mat_sub(M1, M2);
    mat_add_inplace(C22, M3);
    mat_add_inplace(C22, M6);

    /* Free M products */
    mat_free(M1); mat_free(M2); mat_free(M3); mat_free(M4);
    mat_free(M5); mat_free(M6); mat_free(M7);

    /* ---- Assemble C ---- */
    Matrix *C = mat_create(n, n);
    set_submatrix(C, 0,    0,    C11, half);
    set_submatrix(C, 0,    half, C12, half);
    set_submatrix(C, half, 0,    C21, half);
    set_submatrix(C, half, half, C22, half);

    mat_free(C11); mat_free(C12); mat_free(C21); mat_free(C22);

    return C;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/*  Handles arbitrary square matrices by padding to power of 2.        */
/* ------------------------------------------------------------------ */
Matrix *strassen_multiply(const Matrix *A, const Matrix *B, int cutoff)
{
    if (A->cols != B->rows) return NULL;
    if (A->rows != A->cols || B->rows != B->cols) return NULL;

    int orig = A->rows;

    int n = 1;
    while (n < orig) n <<= 1;

    if (cutoff < 1) cutoff = 64;

    if (n == orig) {
        return strassen(A, B, n, cutoff);
    }

    /* Pad A and B with zeros to n×n */
    Matrix *Ap = mat_create(n, n);
    Matrix *Bp = mat_create(n, n);

    for (int i = 0; i < orig; i++) {
        memcpy(&MAT_AT(Ap, i, 0), &MAT_AT(A, i, 0),
               (size_t)orig * sizeof(double));
        memcpy(&MAT_AT(Bp, i, 0), &MAT_AT(B, i, 0),
               (size_t)orig * sizeof(double));
    }

    Matrix *Cp = strassen(Ap, Bp, n, cutoff);
    mat_free(Ap);
    mat_free(Bp);

    /* Extract the top-left orig×orig block */
    Matrix *C = mat_create(orig, orig);
    for (int i = 0; i < orig; i++) {
        memcpy(&MAT_AT(C, i, 0), &MAT_AT(Cp, i, 0),
               (size_t)orig * sizeof(double));
    }
    mat_free(Cp);

    return C;
}
