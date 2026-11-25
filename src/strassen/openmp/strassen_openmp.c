#include "strassen_openmp.h"

#include <limits.h>
#include <omp.h>

enum Quadrant { Q11 = 1, Q12, Q21, Q22 };

typedef struct {
    int first;
    int second; /* Positive index adds, negative index subtracts, zero skips. */
} Operand;

static const struct {
    Operand left;
    Operand right;
} product_operands[] = {
    {{Q11,  Q22}, {Q11,  Q22}}, /* M1 = (A11 + A22) * (B11 + B22) */
    {{Q21,  Q22}, {Q11,    0}}, /* M2 = (A21 + A22) * B11 */
    {{Q11,    0}, {Q12, -Q22}}, /* M3 = A11 * (B12 - B22) */
    {{Q22,    0}, {Q21, -Q11}}, /* M4 = A22 * (B21 - B11) */
    {{Q11,  Q12}, {Q22,    0}}, /* M5 = (A11 + A12) * B22 */
    {{Q21, -Q11}, {Q11,  Q12}}, /* M6 = (A21 - A11) * (B11 + B12) */
    {{Q12, -Q22}, {Q21,  Q22}}, /* M7 = (A12 - A22) * (B21 + B22) */
};

enum { PRODUCT_COUNT = sizeof(product_operands) / sizeof(product_operands[0]) };

static const double *quadrant_start(const Matrix *matrix, int quadrant)
{
    int index = quadrant - 1;
    int half = matrix->rows / 2;
    int row = (index / 2) * half;
    int col = (index % 2) * half;
    return &MAT_AT(matrix, row, col);
}

static Matrix *form_operand(const Matrix *matrix, Operand operand)
{
    int half = matrix->rows / 2;
    Matrix *result = mat_create(half, half);
    if (!result) return NULL;

    const double *first = quadrant_start(matrix, operand.first);
    if (operand.second == 0) {
        for (int i = 0; i < half; i++) {
            memcpy(&MAT_AT(result, i, 0), first + i * matrix->cols,
                   (size_t)half * sizeof(double));
        }
    } else {
        const double *second = quadrant_start(matrix, abs(operand.second));
        double sign = operand.second < 0 ? -1.0 : 1.0;
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                int offset = i * matrix->cols + j;
                MAT_AT(result, i, j) = first[offset] + sign * second[offset];
            }
        }
    }
    return result;
}

static Matrix *combine_products(Matrix *const products[PRODUCT_COUNT])
{
    int half = products[0]->rows;
    Matrix *C = mat_create(2 * half, 2 * half);
    if (!C) return NULL;

    const double *m1 = products[0]->data;
    const double *m2 = products[1]->data;
    const double *m3 = products[2]->data;
    const double *m4 = products[3]->data;
    const double *m5 = products[4]->data;
    const double *m6 = products[5]->data;
    const double *m7 = products[6]->data;
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            int k = i * half + j;
            MAT_AT(C, i, j) = m1[k] + m4[k] - m5[k] + m7[k];
            MAT_AT(C, i, j + half) = m3[k] + m5[k];
            MAT_AT(C, i + half, j) = m2[k] + m4[k];
            MAT_AT(C, i + half, j + half) = m1[k] - m2[k] + m3[k] + m6[k];
        }
    }
    return C;
}

static Matrix *multiply_tasks(const Matrix *A, const Matrix *B, int cutoff,
                              int task_depth)
{
    if (task_depth == 0 || A->rows <= cutoff) {
        return strassen_multiply(A, B, cutoff);
    }

    Matrix *products[PRODUCT_COUNT];
    /* Wait for all seven products before assembling the result. */
    #pragma omp taskgroup
    {
        for (int i = 0; i < PRODUCT_COUNT; i++) {
            #pragma omp task default(none) shared(products, product_operands) \
                firstprivate(A, B, cutoff, task_depth, i)
            {
                Matrix *left = form_operand(A, product_operands[i].left);
                Matrix *right = form_operand(B, product_operands[i].right);
                products[i] = NULL;
                if (left && right) {
                    products[i] = multiply_tasks(left, right, cutoff, task_depth - 1);
                }
                mat_free(left);
                mat_free(right);
            }
        }
    }

    int complete = 1;
    for (int i = 0; i < PRODUCT_COUNT; i++) {
        if (!products[i]) complete = 0;
    }
    Matrix *C = complete ? combine_products(products) : NULL;
    for (int i = 0; i < PRODUCT_COUNT; i++) {
        mat_free(products[i]);
    }
    return C;
}

static Matrix *copy_square(const Matrix *source, int size)
{
    Matrix *copy = mat_create(size, size);
    if (!copy) return NULL;

    int overlap = source->rows < size ? source->rows : size;
    for (int i = 0; i < overlap; i++) {
        memcpy(&MAT_AT(copy, i, 0), &MAT_AT(source, i, 0),
               (size_t)overlap * sizeof(double));
    }
    return copy;
}

Matrix *strassen_multiply_openmp(const Matrix *A, const Matrix *B, int cutoff)
{
    if (!A || !B || !A->data || !B->data || A->rows <= 0 ||
        A->rows != A->cols || B->rows != B->cols || A->rows != B->rows) {
        return NULL;
    }

    if (cutoff < 1) cutoff = 64;
    int size = A->rows;
    if (size > INT_MAX / size) return NULL;
    if (size <= cutoff) return mat_mul_naive(A, B);

    int padded_size = 1;
    while (padded_size < size) padded_size *= 2;
    if (padded_size > INT_MAX / padded_size) return NULL;

    Matrix *padded_a = NULL;
    Matrix *padded_b = NULL;
    if (size != padded_size) {
        padded_a = copy_square(A, padded_size);
        padded_b = copy_square(B, padded_size);
        if (!padded_a || !padded_b) {
            mat_free(padded_a);
            mat_free(padded_b);
            return NULL;
        }
        A = padded_a;
        B = padded_b;
    }

    Matrix *C;
    #pragma omp parallel default(none) shared(A, B, C, cutoff)
    {
        #pragma omp single
        {
            int task_depth = 0;
            int workers = omp_get_num_threads();
            /* Group workers in sevens to count the task levels needed. */
            while (workers > 1) {
                workers = 1 + (workers - 1) / PRODUCT_COUNT;
                task_depth++;
            }
            C = multiply_tasks(A, B, cutoff, task_depth);
        }
    }

    mat_free(padded_a);
    mat_free(padded_b);
    if (C && size != padded_size) {
        Matrix *cropped = copy_square(C, size);
        mat_free(C);
        C = cropped;
    }
    return C;
}
