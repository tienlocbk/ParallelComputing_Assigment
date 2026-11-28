#include "strassen_mpi.h"

// Quadrants are numbered in row-major order. Each row describes M1 through M7.
static const int left_terms[7][4] = {
    {1,0,0,1}, {0,0,1,1}, {1,0,0,0}, {0,0,0,1},
    {1,1,0,0}, {-1,0,1,0}, {0,1,0,-1}
};
static const int right_terms[7][4] = {
    {1,0,0,1}, {1,0,0,0}, {0,1,0,-1}, {-1,0,1,0},
    {0,0,0,1}, {1,1,0,0}, {0,0,1,1}
};
static const int output_terms[7][4] = {
    {1,0,0,1}, {0,0,1,-1}, {0,1,0,1}, {1,0,1,0},
    {-1,1,0,0}, {0,0,0,1}, {1,0,0,0}
};

static Matrix *allocate(int n, MPI_Comm comm) {
    Matrix *m = mat_create(n, n);
    if (!m) {
        fprintf(stderr, "Could not allocate a %d x %d matrix.\n", n, n);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    return m;
}

// The bounds implement zero padding for inputs and cropping for outputs.
static void combine(Matrix *whole, Matrix *block, const int terms[4], bool assemble) {
    int h = block->rows;
    for (int q = 0; q < 4; ++q) {
        if (!terms[q]) continue;
        int row = (q / 2) * h;
        int col = (q % 2) * h;
        for (int i = 0; i < h && row + i < whole->rows; ++i) {
            for (int j = 0; j < h && col + j < whole->cols; ++j) {
                double &cell = MAT_AT(whole, row + i, col + j);
                double &term = MAT_AT(block, i, j);
                if (assemble) cell += terms[q] * term;
                else term += terms[q] * cell;
            }
        }
    }
}

static Matrix *multiply(Matrix *a, Matrix *b, int n, int cutoff, MPI_Comm comm, StrassenKernel kernel) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    if (ranks == 1) return kernel(a, b, cutoff);


    // ------------------ Broadcast inputs from root ------------------
    if (rank != 0) {
        a = allocate(n, comm);
        b = allocate(n, comm);
    }
    MPI_Bcast(a->data, n * n, MPI_DOUBLE, 0, comm);
    MPI_Bcast(b->data, n * n, MPI_DOUBLE, 0, comm);

    // ------------------ Compute the products assigned to this rank ------------------
    Matrix *sum = allocate(n, comm);
    int half = (n + 1) / 2;
    for (int k = rank; k < 7; k += ranks) {
        Matrix *left = allocate(half, comm);
        Matrix *right = allocate(half, comm);
        combine(a, left, left_terms[k], false);
        combine(b, right, right_terms[k], false);
        Matrix *product = kernel(left, right, cutoff);
        if (!product) {
            fprintf(stderr, "Rank %d could not multiply the operands for product %d.\n",
                    rank, k + 1);
            MPI_Abort(comm, EXIT_FAILURE);
        }
        combine(sum, product, output_terms[k], true);
        mat_free(left);
        mat_free(right);
        mat_free(product);
    }
    // ------------------ Reduce partial output matrices to root ------------------
    void *send_buffer = rank == 0 ? MPI_IN_PLACE : sum->data;
    double *root_buffer = rank == 0 ? sum->data : nullptr;
    MPI_Reduce(send_buffer, root_buffer, n * n, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank != 0) {
        mat_free(a);
        mat_free(b);
        mat_free(sum);
        return nullptr;
    }
    return sum;
}

Matrix *strassen_multiply_mpi(Matrix *a, Matrix *b, int cutoff,
                              MPI_Comm comm, StrassenKernel kernel) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // ------------------ Broadcast problem size and cutoff ------------------
    int settings[2] = {0, cutoff};
    if (rank == 0) settings[0] = a->rows;
    MPI_Bcast(settings, 2, MPI_INT, 0, comm);
    int n = settings[0];
    cutoff = settings[1];

    // ------------------ Select ranks for the seven products ------------------
    int active = n <= cutoff ? 1 : 7;
    MPI_Comm work;
    MPI_Comm_split(comm, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
    if (work == MPI_COMM_NULL) return nullptr;
    Matrix *result = multiply(a, b, n, cutoff, work, kernel);
    MPI_Comm_free(&work);
    return result;
}
