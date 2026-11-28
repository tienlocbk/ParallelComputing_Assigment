#include "run_mpi.h"
#include "../algo/strassen.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int status = run_mpi(argc, argv, strassen_multiply, 0);
    MPI_Finalize();
    return status;
}
