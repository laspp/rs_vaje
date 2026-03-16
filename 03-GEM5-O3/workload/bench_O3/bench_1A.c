/*
 * Benchmark 1A — Independent Integer Chain
 * -----------------------------------------
 * Demonstrates ILP exploitation in O3 execution.
 *
 * All additions operate on INDEPENDENT registers.
 * An O3 processor can issue all of them in parallel
 * across its IntALUs in a single cycle.
 *
 * Expected Konata view:
 *   Multiple instructions overlap in the Execute stage.
 *   ROB fills quickly and commits in bursts.
 *
 */

#include <stdint.h>
#include <gem5/m5ops.h>

#define ITERATIONS 1000

/* Prevent compiler from optimising away results */

int main(void) {

    volatile uint64_t sink = 0;
    volatile uint64_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
    volatile uint64_t b0 = 5, b1 = 6, b2 = 7, b3 = 8;
    volatile uint64_t r0, r1, r2, r3;

    
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    
    for (int i = 0; i < ITERATIONS; i++) {
        /*
         * All four additions are FULLY INDEPENDENT of each other.
         * O3 can issue all four to four separate IntALUs in one cycle.
         * An in-order processor must issue them one at a time.
         */
        r0 = a0 * b0;   /* independent */
        r1 = a1 * b1;   /* independent */
        r2 = a2 * b2;   /* independent */
        r3 = a3 * b3;   /* independent */
    }

    sink = r0 ^ r1 ^ r2 ^ r3;

    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    

    return 0;
}
