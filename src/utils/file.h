#pragma once

#include "matrix.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>

static int parse_positive_int(const char *text)
{
    while (isspace((unsigned char)*text)) {
        text++;
    }
    if (*text == '+') {
        text++;
    }

    int value = 0;
    for (; *text != '\0'; text++) {
        if (*text < '0' || *text > '9') {
            return 0;
        }

        int digit = *text - '0';
        /* Check that appending the digit still fits in an int. */
        if (value > (INT_MAX - digit) / 10) {
            return 0;
        }
        value = value * 10 + digit;
    }
    return value;
}

static int check_result(const Matrix *A, const Matrix *B, const Matrix *C)
{
    Matrix *expected = mat_mul_naive(A, B);
    if (!expected) {
        fprintf(stderr, "Could not allocate the reference result.\n");
        return 0;
    }

    int passed = 1;
    for (int i = 0; i < C->rows * C->cols; i++) {
        double actual = C->data[i];
        double reference = expected->data[i];
        double tolerance = 1e-9 * fmax(1.0, fabs(reference));
        if (fabs(actual - reference) <= tolerance) {
            continue;
        }
        fprintf(stderr, "Mismatch at row %d, column %d (got %.17g, expected %.17g).\n",
                i / C->cols, i % C->cols, actual, reference);
        passed = 0;
        break;
    }
    mat_free(expected);
    return passed;
}

static int parse_arguments(int argc, char **argv, int default_cutoff, int *matrix_size,
                           int *strassen_cutoff, int *verify)
{
    int check_requested = argc > 1 && strcmp(argv[argc - 1], "--check") == 0;
    int value_count = argc - 1;
    if (check_requested) {
        value_count--;
    }
    if (value_count != 1 && value_count != 2) {
        fprintf(stderr, "Usage: %s N [cutoff] [--check]\n", argv[0]);
        return 0;
    }

    int size = parse_positive_int(argv[1]);
    int cutoff = default_cutoff;
    if (value_count == 2) {
        cutoff = parse_positive_int(argv[2]);
    }
    if (size == 0 || cutoff == 0) {
        fprintf(stderr, "N and cutoff must be positive integers that fit in an int.\n");
        return 0;
    }
    if (size > INT_MAX / size) {
        fprintf(stderr, "N is too large for the matrix index type.\n");
        return 0;
    }

    *matrix_size = size;
    *strassen_cutoff = cutoff;
    *verify = check_requested;
    return 1;
}

static int read_csv_row(const char *text, double *row, int columns)
{
    for (int col = 0; col < columns; col++) {
        char *end;
        double value = strtod(text, &end);
        if (end == text || !isfinite(value)) {
            return 0;
        }
        row[col] = value;

        while (isspace((unsigned char)*end)) {
            end++;
        }
        if (col == columns - 1) {
            return *end == '\0';
        }
        if (*end != ',') {
            return 0;
        }
        text = end + 1;
    }
    return 0;
}

static Matrix *load_csv_matrix(const char *filename, int size)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open %s.\n", filename);
        return NULL;
    }

    Matrix *matrix = mat_create(size, size);
    char *line = NULL;
    size_t capacity = 0;
    int rows = 0;
    if (matrix) {
        while (rows < size && getline(&line, &capacity, file) >= 0) {
            if (!read_csv_row(line, &MAT_AT(matrix, rows, 0), size)) {
                break;
            }
            rows++;
        }
    }

    if (!matrix) {
        fprintf(stderr, "Could not allocate the matrix for %s.\n", filename);
    } else if (rows != size || getline(&line, &capacity, file) >= 0 || !feof(file)) {
        fprintf(stderr, "%s must contain exactly %d rows of %d comma-separated numbers.\n",
                filename, size, size);
        mat_free(matrix);
        matrix = NULL;
    }

    free(line);
    fclose(file);
    return matrix;
}

