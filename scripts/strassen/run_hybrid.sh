#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
dir="$root/output"
if [[ $# -lt 1 || $# -gt 3 || ! $1 =~ ^[1-9][0-9]{0,4}$ ]] || (( $1 > 46340 )); then
    echo "Usage: $0 N [cutoff] [--check]" >&2
    exit 2
fi
mkdir -p "$dir"
cd "$dir"
if [[ ! -f A.csv || ! -f B.csv ]]; then
    echo "Expected shared output/A.csv and output/B.csv. See the README to generate inputs." >&2
    exit 1
fi
mpicxx -std=c++17 -O3 -fopenmp "$root/src/strassen/hybrid/main.c" \
    "$root/src/strassen/mpi/strassen_mpi.cpp" "$root/src/strassen/algo/strassen.cpp" \
    "$root/src/strassen/openmp/strassen_openmp.c" -o strassen_hybrid
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
exec mpirun --bind-to none -np "${MPI_RANKS:-4}" ./strassen_hybrid "$@"
