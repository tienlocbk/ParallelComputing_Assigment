#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * matrix.h — Simple dense matrix (flat 1D)
 *
 * Storage:  A->data[i * A->cols + j]  =  element at row i, col j
 *
 */

/* ------------------------------------------------------------------ */
/*  Struct                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    int    rows;
    int    cols;
    double *data;   /* flat array, length = rows * cols */
} Matrix;

/* ------------------------------------------------------------------ */
/*  Element access macro                                               */
/* ------------------------------------------------------------------ */

#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/* Allocate a rows×cols matrix, zeroed out. Returns NULL on failure. */
static inline Matrix *mat_create(int rows, int cols)
{
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (!m->data) { free(m); return NULL; }
    return m;
}

/* Free a matrix and its data. Safe to call with NULL. */
static inline void mat_free(Matrix *m)
{
    if (m) { free(m->data); free(m); }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Fill every element with a random value in [0, 1]. */
static inline void mat_randomize(Matrix *m)
{
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++)
        m->data[i] = (double)rand() / RAND_MAX;
}

/* Fill every element with val. */
static inline void mat_fill(Matrix *m, double val)
{
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++)
        m->data[i] = val;
}

/* Deep copy. */
static inline Matrix *mat_copy(const Matrix *src)
{
    Matrix *dst = mat_create(src->rows, src->cols);
    if (dst)
        memcpy(dst->data, src->data, (size_t)src->rows * src->cols * sizeof(double));
    return dst;
}

/* Print */
static inline void mat_print(const Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++)
            printf("%8.3f ", MAT_AT(m, i, j));
        printf("\n");
    }
}

/* ------------------------------------------------------------------ */
/*  Matrix Addition:  C = A + B                                        */
/*  Need same dimensions. Returns NULL on mismatch.   */
/* ------------------------------------------------------------------ */

static inline Matrix *mat_add(const Matrix *A, const Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) return NULL;

    Matrix *C = mat_create(A->rows, A->cols);
    if (!C) return NULL;

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        C->data[i] = A->data[i] + B->data[i];

    return C;
}

/* In-place:  A += B */
static inline int mat_add_inplace(Matrix *A, const Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) return -1;

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] += B->data[i];

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Matrix Subtraction:  C = A - B                                     */
/* ------------------------------------------------------------------ */

static inline Matrix *mat_sub(const Matrix *A, const Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) return NULL;

    Matrix *C = mat_create(A->rows, A->cols);
    if (!C) return NULL;

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        C->data[i] = A->data[i] - B->data[i];

    return C;
}

/* In-place:  A -= B */
static inline int mat_sub_inplace(Matrix *A, const Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) return -1;

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] -= B->data[i];

    return 0;
}

static inline Matrix *mat_mul_naive(const Matrix *A, const Matrix *B)
{
    if (A->cols != B->rows) return NULL;

    int m = A->rows;
    int k = A->cols;
    int n = B->cols;

    Matrix *C = mat_create(m, n);
    if (!C) return NULL;

    /* ikj loop order — better cache behaviour for row-major storage */
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            double a_ip = MAT_AT(A, i, p);
            for (int j = 0; j < n; j++) {
                MAT_AT(C, i, j) += a_ip * MAT_AT(B, p, j);
            }
        }
    }

    return C;
}

#ifdef __cplusplus
/* The caller frees the result with mat_free. Multiplication uses the naive kernel. */
static inline Matrix *operator+(const Matrix &A, const Matrix &B)
{
    return mat_add(&A, &B);
}

static inline Matrix *operator*(const Matrix &A, const Matrix &B)
{
    return mat_mul_naive(&A, &B);
}
#endif

/* ------------------------------------------------------------------ */
/*  Equality check (with tolerance, for correctness testing)           */
/* ------------------------------------------------------------------ */

static inline int mat_equal(const Matrix *A, const Matrix *B, double tol)
{
    if (A->rows != B->rows || A->cols != B->cols) return 0;

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) {
        double diff = A->data[i] - B->data[i];
        if (diff < -tol || diff > tol) return 0;
    }
    return 1;
}
