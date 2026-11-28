#pragma once

#include <math.h>
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
/*  Binary I/O (for cross-testing with Strassen and fast persistence) */
/* ------------------------------------------------------------------ */

/* Save matrix to a binary file. Returns 1 on success, 0 on failure. */
static inline int mat_save_bin(const char *path, const Matrix *m)
{
    if (!path || !m || !m->data) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    int header[2] = {m->rows, m->cols};
    if (fwrite(header, sizeof(int), 2, f) != 2) {
        fclose(f);
        return 0;
    }

    size_t total = (size_t)m->rows * m->cols;
    if (fwrite(m->data, sizeof(double), total, f) != total) {
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

/* Load matrix from a binary file. Returns NULL on failure. */
static inline Matrix *mat_load_bin(const char *path)
{
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int header[2] = {0, 0};
    if (fread(header, sizeof(int), 2, f) != 2) {
        fclose(f);
        return NULL;
    }

    Matrix *m = mat_create(header[0], header[1]);
    if (!m) {
        fclose(f);
        return NULL;
    }

    size_t total = (size_t)m->rows * m->cols;
    if (fread(m->data, sizeof(double), total, f) != total) {
        mat_free(m);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return m;
}

/* Load pair (A, B) from <prefix>_A.bin & <prefix>_B.bin, or generate with seed.
 * If prefix is provided and files do not exist yet, saves them to disk.
 * Returns 1 on success, 0 on failure. */
static inline int mat_load_or_generate_pair(int N, const char *prefix, unsigned int seed,
                                            Matrix **A_out, Matrix **B_out)
{
    Matrix *A = NULL;
    Matrix *B = NULL;
    char pathA[512];
    char pathB[512];
    int has_prefix = (prefix && prefix[0] != '\0');

    if (has_prefix) {
        snprintf(pathA, sizeof(pathA), "%s_A.bin", prefix);
        snprintf(pathB, sizeof(pathB), "%s_B.bin", prefix);

        A = mat_load_bin(pathA);
        B = mat_load_bin(pathB);

        if (A && B) {
            if (A->rows != N || A->cols != N || B->rows != N || B->cols != N) {
                fprintf(stderr, "Error: Binary matrix dimensions do not match N=%d.\n", N);
                mat_free(A);
                mat_free(B);
                return 0;
            }
            *A_out = A;
            *B_out = B;
            return 1;
        }
        mat_free(A);
        mat_free(B);
    }

    srand(seed);
    A = mat_create(N, N);
    B = mat_create(N, N);
    if (!A || !B) {
        mat_free(A);
        mat_free(B);
        return 0;
    }
    mat_randomize(A);
    mat_randomize(B);

    if (has_prefix) {
        if (!mat_save_bin(pathA, A) || !mat_save_bin(pathB, B)) {
            fprintf(stderr, "Warning: Failed to save generated input binary files.\n");
        }
    }

    *A_out = A;
    *B_out = B;
    return 1;
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

/* ------------------------------------------------------------------ */
/*  Freivalds' algorithm for O(trials * N^2) verification              */
/*  Verifies A * B == C with failure probability <= 2^(-trials)       */
/*  Suitable for large matrices (N = 1000 to 10000)                   */
/*  Returns 1 on PASS, 0 on FAIL.                                     */
/* ------------------------------------------------------------------ */

static inline int mat_verify_freivalds(const Matrix *A, const Matrix *B, const Matrix *C,
                                       int trials, double tol)
{
    if (!A || !B || !C) return 0;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return 0;

    int m = A->rows;
    int k = A->cols;
    int n = B->cols;

    if (trials <= 0) trials = 5;
    if (tol <= 0.0) tol = 1e-9;

    double *r   = (double *)malloc((size_t)n * sizeof(double));
    double *br  = (double *)malloc((size_t)k * sizeof(double));
    double *abr = (double *)malloc((size_t)m * sizeof(double));
    double *cr  = (double *)malloc((size_t)m * sizeof(double));

    if (!r || !br || !abr || !cr) {
        free(r); free(br); free(abr); free(cr);
        return 0;
    }

    int pass = 1;

    for (int t = 0; t < trials; t++) {
        /* Vector r ngau nhien thuoc {-1, 1}^n */
        for (int j = 0; j < n; j++) {
            r[j] = (rand() & 1) ? 1.0 : -1.0;
        }

        /* br = B * r (kich thuoc k) */
        for (int p = 0; p < k; p++) {
            double sum = 0.0;
            const double *b_row = &MAT_AT(B, p, 0);
            for (int j = 0; j < n; j++) {
                sum += b_row[j] * r[j];
            }
            br[p] = sum;
        }

        /* abr = A * (B * r) (kich thuoc m) */
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            const double *a_row = &MAT_AT(A, i, 0);
            for (int p = 0; p < k; p++) {
                sum += a_row[p] * br[p];
            }
            abr[i] = sum;
        }

        /* cr = C * r (kich thuoc m) */
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            const double *c_row = &MAT_AT(C, i, 0);
            for (int j = 0; j < n; j++) {
                sum += c_row[j] * r[j];
            }
            cr[i] = sum;
        }

        /* Kiem tra sai so tuong doi giua abr va cr */
        double max_diff = 0.0;
        double max_norm = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = fabs(abr[i] - cr[i]);
            double norm = fabs(abr[i]);
            if (diff > max_diff) max_diff = diff;
            if (norm > max_norm) max_norm = norm;
        }

        double rel_err = max_diff / (max_norm + 1e-12);
        if (rel_err > tol) {
            pass = 0;
            break;
        }
    }

    free(r);
    free(br);
    free(abr);
    free(cr);

    return pass;
}