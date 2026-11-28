#include "matrix.h"
#include "timer.h"
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

void matmul_hybrid_local_ikj(const Matrix* A_local, const Matrix* B, Matrix* C_local) {
    int rows = A_local->rows;
    int k    = A_local->cols;
    int n    = B->cols;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        double* c_row = &MAT_AT(C_local, i, 0);
        const double* a_row = &MAT_AT(A_local, i, 0);
        for (int p = 0; p < k; p++) {
            double a_ip = a_row[p];
            const double* b_row = &MAT_AT(B, p, 0);
            for (int j = 0; j < n; j++) {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED && rank == 0) {
        fprintf(stderr, "Warning: MPI does not guarantee MPI_THREAD_FUNNELED.\n");
    }

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: mpirun -np <P> %s <N> <threads_per_rank> [verify:0|1] [bin_prefix]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const int threads_per_rank = atoi(argv[2]);
    const int do_verify = (argc >= 4) ? atoi(argv[3]) : 0;
    const char *bin_prefix = (argc >= 5) ? argv[4] : "";

    if (N <= 0 || threads_per_rank <= 0 || (long long)N * N > INT_MAX) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid N or threads_per_rank.\n");
        }
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(threads_per_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total_start = timer_now();

    vector<int> numEle(size), Start(size);
    int temp = N / size;
    int rem  = N % size;
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int row = temp + (i < rem ? 1 : 0);
        numEle[i] = row * N;
        Start[i] = offset * N;
        offset += row;
    }

    const int local_rows  = temp + (rank < rem ? 1 : 0);
    const int local_elems = numEle[rank];

    Matrix* A = NULL;
    Matrix* B = NULL;
    Matrix* C = NULL;
    int local_check = 1;

    if (rank == 0) {
        if (!mat_load_or_generate_pair(N, bin_prefix, 42, &A, &B)) {
            local_check = 0;
        } 
        else {
            C = mat_create(N, N);
            if (!C) local_check = 0;
        }
    } 
    else {
        B = mat_create(N, N);
        if (!B) local_check = 0;
    }

    Matrix* A_local = mat_create(local_rows > 0 ? local_rows : 1, N);
    Matrix* C_local = mat_create(local_rows > 0 ? local_rows : 1, N);
    if (!A_local || !C_local) {
        local_check = 0;
    } 
    else {
        A_local->rows = local_rows;
        C_local->rows = local_rows;
    }

    int all_check = 1;
    MPI_Allreduce(&local_check, &all_check, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_check) {
        if (rank == 0) fprintf(stderr, "Error: Initialization or memory allocation failed.\n");
        mat_free(A); mat_free(B); mat_free(C);
        mat_free(A_local); mat_free(C_local);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_comm1_start = timer_now();
    MPI_Scatterv(A ? A->data : NULL, numEle.data(), Start.data(), MPI_DOUBLE,
                 A_local->data, local_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B->data, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double t_comm1_end = timer_now();

    const double t_comp_start = timer_now();
    matmul_hybrid_local_ikj(A_local, B, C_local);
    const double t_comp_end = timer_now();

    const double t_comm2_start = timer_now();
    MPI_Gatherv(C_local->data, local_elems, MPI_DOUBLE,
                C ? C->data : NULL, numEle.data(), Start.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    const double t_comm2_end = timer_now();

    if (rank == 0 && bin_prefix[0] != '\0') {
        char pathC[512];
        snprintf(pathC, sizeof(pathC), "%s_C_hybrid.bin", bin_prefix);
        if (!mat_save_bin(pathC, C)) {
            fprintf(stderr, "Warning: Failed to save %s.\n", pathC);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total_end = timer_now();

    const double local_comm = (t_comm1_end - t_comm1_start) + (t_comm2_end - t_comm2_start);
    const double local_comp = (t_comp_end - t_comp_start);
    double max_comm = 0.0, max_comp = 0.0;

    MPI_Reduce(&local_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("MODE=HYBRID,N=%d,PROCS=%d,THREADS=%d,COMM_TIME=%.6f,COMP_TIME=%.6f,TOTAL_TIME=%.6f\n",
               N, size, threads_per_rank, max_comm, max_comp, t_total_end - t_total_start);

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
    }

    mat_free(A);
    mat_free(B);
    mat_free(C);
    mat_free(A_local);
    mat_free(C_local);

    MPI_Finalize();
    return exit_code;
}