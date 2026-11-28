# Parallel Computing Assignment

## Shared Strassen inputs

All three runners read `output/A.csv` and `output/B.csv`. Generate the pair
once from the repository root, then pass the same matrix size to each runner.

```bash
(cd output && python3 ../src/strassen/openmp/generate_csv.py 4)
```

The generator replaces both files. You can also supply your own CSVs with
exactly `N` rows and `N` comma-separated numbers per row. The runners build
their executables in `output/` and reuse the CSVs there.

## OpenMP Strassen

From the repository root, run:

```bash
OMP_NUM_THREADS=4 ./scripts/strassen/run_openmp.sh 4 1 --check
```

| Argument | Meaning |
| --- | --- |
| `N` | Required row and column count for both matrices. |
| `cutoff` | Optional positive size at which Strassen switches to naive multiplication. Defaults to `512`. |
| `--check` | Optional correctness check. Place it last. |
| `OMP_NUM_THREADS` | Sets the OpenMP thread count. If unset, the OpenMP runtime chooses the default. |

With more than one thread, OpenMP creates tasks for the seven top-level
products when the input exceeds the cutoff. Each task runs its local
recursion serially.

```bash
cd output
OMP_NUM_THREADS=2 ./strassen_openmp 4 1 --check
```

## MPI Strassen

Install an MPI C++ compiler wrapper and launcher. On Ubuntu or Debian, install
`openmpi-bin libopenmpi-dev g++ python3`. With `mpicxx` and `mpirun` on your
PATH, run this from the repository root:

```bash
MPI_RANKS=2 ./scripts/strassen/run_mpi.sh 4 1 --check
```

The arguments match the OpenMP runner. The default is seven ranks. Set
`MPI_RANKS` to use fewer ranks when CPU slots or memory are limited. Rank 0 reads the
CSVs and broadcasts both matrices to the active ranks. They divide the seven
top-level products round-robin, run the serial Strassen kernel locally, and
sum their contributions on rank 0. Two ranks compute four and three products
respectively. At most seven ranks participate. A single rank or a size at or
below the cutoff uses the serial kernel. Each active rank needs memory for
both inputs, its partial output, and temporary matrices.

### Run across two laptops with Open MPI

Use the same Open MPI version and absolute repository path on both laptops.
Connect them to the same LAN. On Ubuntu or Debian, install `openssh-server`
and run `sudo systemctl enable --now ssh` on each. For matching usernames,
run `ssh-copy-id PEER_IP` from each laptop using its peer's LAN address
(generate a key with `ssh-keygen -t ed25519` first if needed). Verify
`ssh -o BatchMode=yes PEER_IP hostname` works in both directions.

Run the local script above once on each laptop to build its executable.
On the first laptop, create a file named `hosts` in the repository root,
replacing these addresses with the actual first and second laptop addresses:

```text
192.168.1.21 slots=1
192.168.1.22 slots=1
```

From that root, verify that the first command prints both hostnames, then
launch the multiplication. Only rank 0 reads the input files.

```bash
mpirun --hostfile hosts -np 2 --map-by ppr:1:node hostname
run_dir="$PWD/output"
mpirun --hostfile hosts -np 2 --map-by ppr:1:node \
    --wdir "$run_dir" "$run_dir/strassen_mpi" 4 1 --check
```

For whole-program benchmarks, prefix the final `mpirun` command with `time`
and omit `--check`. The program's `multiplication_seconds` includes operand
communication, local multiplication, and result collection.

## Hybrid MPI and OpenMP Strassen

From the repository root, with Open MPI and an OpenMP-capable C++ compiler,
run:

```bash
(cd output && python3 ../src/strassen/openmp/generate_csv.py 256)
MPI_RANKS=4 OMP_NUM_THREADS=4 ./scripts/strassen/run_hybrid.sh 256 32 --check
```

Both settings default to four. The example regenerates the shared inputs at size 256.

An MPI rank is a process. A node is a physical machine. This local run puts
four ranks on one laptop. Each rank can use a four-thread OpenMP team,
including the rank's main thread, for its assigned Strassen products. That
gives up to 16 OpenMP threads across all four ranks. The OS schedules the
threads on available logical CPUs. The local launcher disables MPI binding
so an OpenMP team can use more than one CPU.

With four ranks, the seven top-level products are assigned as 2, 2, 2, and 1.
The cutoff of 32 in the example keeps the rank-local products large enough
to create OpenMP tasks. For a multi-laptop run, use the hostfile instructions
above and launch `strassen_hybrid` from its output directory. The mapping
`--map-by ppr:1:node:PE=4 --bind-to core` gives each laptop one rank and
assigns four cores to that rank when four cores are available. Set
`OMP_NUM_THREADS=4` on each rank, for example with Open MPI's
`-x OMP_NUM_THREADS=4`.
