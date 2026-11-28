#pragma once

/*
 * timer.h — Unified wall-clock timer
 *
 * Priority:
 *   1. MPI_Wtime()          if compiled with MPI  (-DUSE_MPI)
 *   2. omp_get_wtime()      if compiled with OpenMP (-fopenmp)
 *   3. clock_gettime()      POSIX fallback (nanosecond resolution)
 *
 * Usage:
 *   double t0 = timer_now();
 *   // ... work ...
 *   double elapsed = timer_now() - t0;   // seconds
 */

#if defined(USE_MPI)
    #include <mpi.h>
    static inline double timer_now() { return MPI_Wtime(); }

#elif defined(_OPENMP)
    #include <omp.h>
    static inline double timer_now() { return omp_get_wtime(); }

#else
    #include <time.h>
    static inline double timer_now() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
