#include "strassen_openmp.h"

#include <omp.h>

static Matrix *multiply_tasks(const Matrix *a, const Matrix *b, int padded_size,
                              int cutoff, StrassenLeaf leaf, void *context)
{
    Matrix *output = mat_create(a->rows, b->cols);
    if (!output) return NULL;

    int half = padded_size / 2;
    int failed = 0;
    omp_lock_t output_lock;
    omp_init_lock(&output_lock);
    #pragma omp taskgroup
    {
        for (int k = 0; k < STRASSEN_PRODUCTS; k++) {
            #pragma omp task default(none) shared(output, output_lock, failed) \
                firstprivate(a, b, half, cutoff, k, leaf, context)
            {
                Matrix *left = mat_create(half, half);
                Matrix *right = mat_create(half, half);
                Matrix *product = NULL;
                if (left && right) {
                    strassen_form_operands(a, b, k, left, right);
                    product = strassen_multiply_with_leaf(left, right, cutoff,
                                                          leaf, context);
                }
                mat_free(left);
                mat_free(right);

                if (product) {
                    omp_set_lock(&output_lock);
                    strassen_accumulate(output, product, k);
                    omp_unset_lock(&output_lock);
                    mat_free(product);
                } else {
                    #pragma omp atomic write
                    failed = 1;
                }
            }
        }
    }
    omp_destroy_lock(&output_lock);
    if (failed) {
        mat_free(output);
        return NULL;
    }
    return output;
}

Matrix *strassen_multiply_openmp(const Matrix *a, const Matrix *b, int cutoff)
{
    return strassen_multiply_openmp_with_leaf(a, b, cutoff, NULL, NULL);
}

Matrix *strassen_multiply_openmp_with_leaf(const Matrix *a, const Matrix *b,
                                           int cutoff, StrassenLeaf leaf,
                                           void *context)
{
    int size = strassen_size(a, b);
    if (!size) return NULL;
    if (cutoff < 1) cutoff = STRASSEN_DEFAULT_CUTOFF;
    if (size <= cutoff)
        return strassen_multiply_with_leaf(a, b, cutoff, leaf, context);

    int padded = strassen_padded_size(size);
    if (!padded) return NULL;

    Matrix *output;
    #pragma omp parallel default(none) shared(a, b, output, cutoff, padded, leaf, context)
    {
        #pragma omp single
        {
            output = omp_get_num_threads() == 1
                ? strassen_multiply_with_leaf(a, b, cutoff, leaf, context)
                : multiply_tasks(a, b, padded, cutoff, leaf, context);
        }
    }
    return output;
}
