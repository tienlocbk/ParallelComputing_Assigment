#include "matrix.h"
#include "timer.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void matmul_omp_ikj(const Matrix *A, const Matrix *B, Matrix *C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double* c_row = &MAT_AT(C, i, 0);
        const double* a_row = &MAT_AT(A, i, 0);
        for (int p = 0; p < k; p++) {
            const double a_ip = a_row[p];
            const double* b_row = &MAT_AT(B, p, 0);
            for (int j = 0; j < n; j++) {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <num_threads> [verify:0|1] [bin_prefix]\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    const int num_threads = atoi(argv[2]);
    const int do_verify = (argc >= 4) ? atoi(argv[3]) : 0;
    const char *bin_prefix = (argc >= 5) ? argv[4] : "";

    if (N <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error: N and num_threads must be positive.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    const double t_total_start = timer_now();

    Matrix *A = NULL;
    Matrix *B = NULL;
    if (!mat_load_or_generate_pair(N, bin_prefix, 42, &A, &B)) {
        fprintf(stderr, "Error: Failed to initialize input matrices.\n");
        return 1;
    }

    Matrix *C = mat_create(N, N);
    if (!C) {
        fprintf(stderr, "Error: Failed to allocate output matrix C.\n");
        mat_free(A);
        mat_free(B);
        return 1;
    }

    const double t_comp_start = timer_now();
    matmul_omp_ikj(A, B, C);
    const double t_comp_end = timer_now();

    if (bin_prefix[0] != '\0') {
        char pathC[512];
        snprintf(pathC, sizeof(pathC), "%s_C_omp.bin", bin_prefix);
        if (!mat_save_bin(pathC, C)) {
            fprintf(stderr, "Warning: Failed to save %s.\n", pathC);
        }
    }

    const double t_total_end = timer_now();

    printf("MODE=OMP,N=%d,THREADS=%d,COMP_TIME=%.6f,TOTAL_TIME=%.6f\n",
           N, num_threads, t_comp_end - t_comp_start, t_total_end - t_total_start);

    int exit_code = 0;
    if (do_verify) {
        int ok = 1;
        if (N <= 512) {
            Matrix *C_ref = mat_mul_naive(A, B);
            const int eq = C_ref ? mat_equal(C, C_ref, 1e-9) : 0;
            printf("[Verify Exact vs mat_mul_naive] %s\n", eq ? "PASS" : "FAIL");
            mat_free(C_ref);
            ok = ok && eq;
        }
        srand(2026);
        const int fv = mat_verify_freivalds(A, B, C, 5, 1e-9);
        printf("[Verify Freivalds x5] %s\n", fv ? "PASS" : "FAIL");
        ok = ok && fv;
        if (!ok) exit_code = 2;
    }

    mat_free(A);
    mat_free(B);
    mat_free(C);
    return exit_code;
}