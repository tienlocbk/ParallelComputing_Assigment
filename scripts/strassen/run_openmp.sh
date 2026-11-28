#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 N [cutoff] [--check]"
}

if [[ $# -eq 1 && $1 == --help ]]; then
    usage
    exit 0
fi

if [[ ${!#} == --check && $# -gt 0 ]]; then
    check=(--check)
    set -- "${@:1:$#-1}"
else
    check=()
fi

# Validate arguments before building or generating input files.
if [[ $# -lt 1 || $# -gt 2 || ! ${1:-} =~ ^[1-9][0-9]{0,4}$ ||
      ! ${2-1} =~ ^[1-9][0-9]{0,9}$ ]]; then
    usage >&2
    exit 2
fi

size=$1
cutoff=${2-0}
if (( size > 46340 || cutoff > 2147483647 )); then
    echo "N or cutoff exceeds the matrix runner's integer limits." >&2
    exit 2
fi

if (( size > 32768 && size > cutoff )); then
    echo "N requires padding beyond the matrix runner's integer limits." >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="$repo_dir/output"
source_dir="$repo_dir/src/strassen/openmp"
mkdir -p -- "$run_dir"
cd -- "$run_dir"
if [[ ! -f A.csv || ! -f B.csv ]]; then
    echo "Expected shared output/A.csv and output/B.csv. See the README to generate inputs." >&2
    exit 1
fi

"${CXX:-g++}" -std=c++17 -O3 -fopenmp \
    "$source_dir/main.c" "$source_dir/strassen_openmp.c" \
    "$repo_dir/src/strassen/algo/strassen.cpp" \
    -o "$run_dir/strassen_openmp"

exec ./strassen_openmp "$@" "${check[@]}"
