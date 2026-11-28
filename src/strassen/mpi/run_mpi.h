#pragma once

#include "strassen_mpi.h"
#include "../algo/strassen.h"
#include "../../utils/file.h"

static int run_mpi(int argc, char **argv, StrassenKernel kernel, int threads_per_rank)
{
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ------------------ Read inputs on root ------------------
    int n = 0, cutoff = 0, verify = 0, ok = 1;
    Matrix *a = nullptr, *b = nullptr;
    if (world_rank == 0) {
        ok = parse_arguments(argc, argv, STRASSEN_DEFAULT_CUTOFF, &n, &cutoff, &verify);
        if (ok) {
            a = load_csv_matrix("A.csv", n);
            b = load_csv_matrix("B.csv", n);
            ok = a && b;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ok) {
        // ------------------ Time the distributed multiplication ------------------
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        Matrix *c = strassen_multiply_mpi(a, b, cutoff, MPI_COMM_WORLD, kernel);

        // ------------------ Report and verify on root ------------------
        if (world_rank == 0) {
            if (threads_per_rank > 0) {
                printf("threads_per_rank=%d\n", threads_per_rank);
            }
            printf("N=%d ranks=%d cutoff=%d multiplication_seconds=%.6f\n",
                   n, world_size, cutoff, MPI_Wtime() - start);
            ok = c != nullptr;
            if (verify) {
                ok = ok && check_result(a, b, c);
                printf("Correctness check: %s\n", ok ? "PASS" : "FAIL");
            }
            mat_free(c);
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    mat_free(a);
    mat_free(b);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
