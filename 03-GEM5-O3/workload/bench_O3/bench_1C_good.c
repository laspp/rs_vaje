/*
 * Benchmark 1C — Load-Use Latency Hiding
 * ----------------------------------------
 * Demonstrates how O3 hides memory latency by executing
 * independent arithmetic while waiting for loads.
 *
 * Two versions in one file, selected by #define:
 *   VERSION_GOOD  — arithmetic interleaved, independent of load result
 *   VERSION_BAD   — every arithmetic op immediately uses the load result
 *
 * Compile good version:
 *   riscv64-unknown-elf-gcc -O1 -DVERSION_GOOD -o bench_1c_good bench_1c_load_use.c
 *
 * Compile bad version:
 *   riscv64-unknown-elf-gcc -O1 -DVERSION_BAD  -o bench_1c_bad  bench_1c_load_use.c
 *
 * Expected Konata view (good):
 *   Load and arithmetic instructions overlap in execute.
 *   The load latency is hidden inside the arithmetic work.
 *
 * Expected Konata view (bad):
 *   Arithmetic instructions stall at issue waiting for load results.
 *   Pipeline bubbles visible between load writeback and dependent issue.
 */

#include <stdint.h>
#include <stdlib.h>
#include <gem5/m5ops.h>

#define N 256
#define ITERATIONS 200
//#define VERSION_GOOD 1

volatile uint64_t sink;

int main(void) {
    /* Heap-allocate array so addresses are not compile-time constants */
    volatile uint64_t *arr = (uint64_t *)malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) arr[i] = (uint64_t)(i + 1);

    volatile uint64_t acc  = 0;
    volatile uint64_t side = 1;   /* independent accumulator for the good version */

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < N; i++) {
            /*
             * GOOD: issue the load, then do independent arithmetic
             * on 'side' (unrelated to the load result) while the
             * cache access completes. O3 overlaps the load with
             * the side computation.
             */
            volatile uint64_t loaded = arr[i];          /* load issued */
            side = side * 6364136223846793005ULL + 1442695040888963407ULL; /* independent work */
            side = side ^ (side >> 33);
            acc += loaded;                     /* use load result only here */
        }
    }

    sink = acc ^ side;
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    free(arr);
    return 0;
}
