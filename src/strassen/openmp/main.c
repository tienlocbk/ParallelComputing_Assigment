#define _POSIX_C_SOURCE 200809L

#include "strassen_openmp.h"
#include "../../utils/timer.h"

#include "../../utils/file.h"

int main(int argc, char **argv)
{
    int size;
    int cutoff;
    int check_requested;
    if (!parse_arguments(argc, argv, STRASSEN_DEFAULT_CUTOFF,
                         &size, &cutoff, &check_requested)) {
        return EXIT_FAILURE;
    }

    Matrix *A = load_csv_matrix("A.csv", size);
    if (!A) {
        return EXIT_FAILURE;
    }
    Matrix *B = load_csv_matrix("B.csv", size);
    if (!B) {
        mat_free(A);
        return EXIT_FAILURE;
    }

    double start = timer_now();
    Matrix *C = strassen_multiply_openmp(A, B, cutoff);
    double elapsed = timer_now() - start;

    if (!C) {
        fprintf(stderr, "Could not multiply the matrices.\n");
        mat_free(A);
        mat_free(B);
        return EXIT_FAILURE;
    }

    printf("N=%d cutoff=%d multiplication_seconds=%.6f\n", size, cutoff, elapsed);
    int passed = 1;
    if (check_requested) {
        passed = check_result(A, B, C);
        printf("Correctness check: %s\n", passed ? "PASS" : "FAIL");
    }

    mat_free(A);
    mat_free(B);
    mat_free(C);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
