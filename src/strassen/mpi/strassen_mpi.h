#pragma once

#include <mpi.h>
#include "../../utils/matrix.h"

using StrassenKernel = Matrix *(*)(const Matrix *, const Matrix *, int);

// Collective on comm. Rank 0 supplies equally sized square matrices and a
// positive cutoff. Other ranks may pass nullptr for both matrices.
// Each rank supplies its local kernel. Rank 0 owns the returned matrix.
Matrix *strassen_multiply_mpi(Matrix *a, Matrix *b, int cutoff,
                              MPI_Comm comm, StrassenKernel kernel);
