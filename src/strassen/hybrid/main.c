#include <omp.h>
#include "../mpi/run_mpi.h"
#include "../openmp/strassen_openmp.h"

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Hybrid Strassen requires MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int status = run_mpi(argc, argv, strassen_multiply_openmp, omp_get_max_threads());
    MPI_Finalize();
    return status;
}
