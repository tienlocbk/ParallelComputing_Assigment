CXX      = g++
MPICXX   = mpic++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra

HEADERS = binary_io.h verify.h ../utils/matrix.h ../utils/timer.h

.PHONY: all omp mpi hybrid clean

all: omp mpi hybrid

omp: matmul_omp
mpi: matmul_mpi
hybrid: matmul_hybrid

matmul_omp: openMP.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

matmul_mpi: MPI.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI $< -o $@

matmul_hybrid: Hybrid.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -fopenmp $< -o $@

clean:
	rm -f matmul_omp matmul_mpi matmul_hybrid *.bin