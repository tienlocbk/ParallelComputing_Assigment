#!/bin/bash
#SBATCH --job-name=strassen
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=strassen-%j.out

set -eo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load openmpi

cd "$SLURM_SUBMIT_DIR/ParallelComputing_Assigment"
work=$(mktemp -d "$PWD/output-job-${SLURM_JOB_ID}-XXXXXX")
cd "$work"
echo "Results directory: $PWD"
scontrol show hostnames "$SLURM_JOB_NODELIST"
mpicxx -O3 -std=c++17 -fopenmp ../src/strassen/{hybrid/main.c,mpi/strassen_mpi.cpp,algo/strassen.cpp,openmp/strassen_openmp.c} -o hybrid
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
python3 ../src/strassen/openmp/generate_csv.py 256
export OMP_NUM_THREADS=4
mpirun -np 4 --map-by ppr:2:node:PE=4 --bind-to core --report-bindings ./hybrid 256 32 --check

size=10000
python3 ../src/strassen/openmp/generate_csv.py "$size"

for trial in 1 2 3; do
    for cutoff in 64 128 256 512; do
        echo "N=$size ranks=4 threads=$OMP_NUM_THREADS cutoff=$cutoff trial=$trial"
        mpirun -np 4 --map-by ppr:2:node:PE=4 --bind-to core ./hybrid "$size" "$cutoff"
    done
done
