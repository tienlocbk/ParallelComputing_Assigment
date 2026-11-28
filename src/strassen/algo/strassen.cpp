#include "strassen.h"

#include <limits.h>

enum StrassenQuadrant { Q11 = 1, Q12, Q21, Q22 };

typedef struct {
    int first;
    int second; /* Zero skips the second quadrant, a negative index subtracts it. */
} StrassenOperand;

static const struct {
    StrassenOperand left;
    StrassenOperand right;
    int output[4]; /* Coefficients for C11, C12, C21, and C22. */
} strassen_products[STRASSEN_PRODUCTS] = {
    {{Q11,  Q22}, {Q11,  Q22}, { 1, 0, 0, 1}},
    {{Q21,  Q22}, {Q11,    0}, { 0, 0, 1,-1}},
    {{Q11,    0}, {Q12, -Q22}, { 0, 1, 0, 1}},
    {{Q22,    0}, {Q21, -Q11}, { 1, 0, 1, 0}},
    {{Q11,  Q12}, {Q22,    0}, {-1, 1, 0, 0}},
    {{Q21, -Q11}, {Q11,  Q12}, { 0, 0, 0, 1}},
    {{Q12, -Q22}, {Q21,  Q22}, { 1, 0, 0, 0}},
};

int strassen_size(const Matrix *a, const Matrix *b)
{
    if (!a || !b || !a->data || !b->data || a->rows <= 0 ||
        a->rows != a->cols || b->rows != b->cols || a->rows != b->rows ||
        a->rows > INT_MAX / a->rows) {
        return 0;
    }
    return a->rows;
}

int strassen_padded_size(int size)
{
    int padded = 1;
    while (padded < size) padded *= 2;
    return padded <= INT_MAX / padded ? padded : 0;
}

/* Copy one quadrant or add it to the destination. Entries beyond the source
 * bounds are logically zero. Copy mode also initializes that padded region. */
static inline void strassen_read_quadrant(const Matrix *matrix, int quadrant,
                                          int copy, Matrix *block)
{
    int half = block->rows;
    int index = abs(quadrant) - 1;
    int row = (index / 2) * half;
    int col = (index % 2) * half;
    int rows = matrix->rows - row;
    int cols = matrix->cols - col;
    if (rows > half) rows = half;
    if (cols > half) cols = half;
    if (rows < 0 || cols <= 0) rows = 0;

    double sign = quadrant < 0 ? -1.0 : 1.0;
    for (int i = 0; i < rows; i++) {
        const double *source = &MAT_AT(matrix, row + i, col);
        double *dest = &MAT_AT(block, i, 0);
        if (copy) {
            memcpy(dest, source, (size_t)cols * sizeof(double));
            memset(dest + cols, 0, (size_t)(half - cols) * sizeof(double));
        } else {
            for (int j = 0; j < cols; j++) {
                dest[j] += sign * source[j];
            }
        }
    }
    if (copy && rows < half)
        memset(block->data + (size_t)rows * half, 0,
               (size_t)(half - rows) * half * sizeof(double));
}

static inline void strassen_form_operand(const Matrix *matrix,
                                         StrassenOperand operand, Matrix *block)
{
    strassen_read_quadrant(matrix, operand.first, 1, block);
    if (operand.second)
        strassen_read_quadrant(matrix, operand.second, 0, block);
}

void strassen_form_operands(const Matrix *a, const Matrix *b, int index,
                            Matrix *left, Matrix *right)
{
    strassen_form_operand(a, strassen_products[index].left, left);
    strassen_form_operand(b, strassen_products[index].right, right);
}

void strassen_accumulate(Matrix *output, const Matrix *product, int index)
{
    int half = product->rows;
    for (int q = 0; q < 4; q++) {
        int sign = strassen_products[index].output[q];
        int row = (q / 2) * half;
        int col = (q % 2) * half;
        int cols = output->cols - col;
        if (!sign || cols <= 0) continue;
        if (cols > half) cols = half;
        for (int i = 0; i < half && row + i < output->rows; i++) {
            double *dest = &MAT_AT(output, row + i, col);
            const double *source = &MAT_AT(product, i, 0);
            for (int j = 0; j < cols; j++)
                dest[j] += sign * source[j];
        }
    }
}

/* Each level reserves two operand blocks and one product block. Siblings
 * reuse these blocks, and deeper calls use the rest of the workspace. */
static size_t workspace_entries(int size, int cutoff)
{
    size_t entries = 0;
    while (size > cutoff) {
        size /= 2;
        entries += 3 * (size_t)size * size;
    }
    return entries;
}

static bool multiply(const Matrix *a, const Matrix *b, Matrix *output,
                     int padded_size, int cutoff, double *workspace,
                     StrassenLeaf leaf, void *context)
{
    if (leaf && padded_size <= cutoff) return leaf(a, b, output, context);
    mat_fill(output, 0.0);
    if (padded_size <= cutoff) {
        mat_mul_naive_into(a, b, output);
        return true;
    }

    int half = padded_size / 2;
    size_t entries = (size_t)half * half;
    Matrix left = {half, half, workspace};
    Matrix right = {half, half, workspace + entries};
    Matrix product = {half, half, workspace + 2 * entries};
    double *child_workspace = workspace + 3 * entries;
    for (int k = 0; k < STRASSEN_PRODUCTS; k++) {
        strassen_form_operands(a, b, k, &left, &right);
        if (!multiply(&left, &right, &product, half, cutoff, child_workspace,
                      leaf, context)) return false;
        strassen_accumulate(output, &product, k);
    }
    return true;
}

Matrix *strassen_multiply(const Matrix *a, const Matrix *b, int cutoff)
{
    return strassen_multiply_with_leaf(a, b, cutoff, NULL, NULL);
}

Matrix *strassen_multiply_with_leaf(const Matrix *a, const Matrix *b, int cutoff,
                                    StrassenLeaf leaf, void *context)
{
    int size = strassen_size(a, b);
    if (!size) return NULL;
    if (cutoff < 1) cutoff = STRASSEN_DEFAULT_CUTOFF;
    if (size <= cutoff && !leaf) return mat_mul_naive(a, b);

    int padded = size <= cutoff ? size : strassen_padded_size(size);
    if (!padded) return NULL;

    Matrix *output = mat_create(size, size);
    if (!output) return NULL;
    size_t entries = workspace_entries(padded, cutoff);
    double *workspace = entries ? (double *)calloc(entries, sizeof(double)) : NULL;
    if (entries && !workspace) {
        mat_free(output);
        return NULL;
    }

    bool ok = multiply(a, b, output, padded, cutoff, workspace, leaf, context);
    free(workspace);
    if (!ok) {
        mat_free(output);
        return NULL;
    }
    return output;
}
