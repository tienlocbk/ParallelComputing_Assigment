# Parallel Computing Assignment

## OpenMP Strassen

From the repository root, build and run with generated CSV inputs.

```sh
g++ -std=c++17 -O3 -fopenmp src/strassen/openmp/main.c \
    src/strassen/openmp/strassen_openmp.c src/strassen/algo/strassen.cpp \
    -o output/strassen_openmp
cd output
python3 ../src/strassen/openmp/generate_csv.py 4
OMP_NUM_THREADS=4 ./strassen_openmp 4 1 --check
```

Arguments are `N [cutoff] [--check]`. The runner reads `A.csv` and `B.csv` from the current directory.
