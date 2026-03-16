/*
 * Benchmark 1B-DEP — Dependent Integer Chain
 * -------------------------------------------
 * Demonstrates how a carried dependency chain limits ILP.
 *
 * Each result feeds directly into the next operation.
 * Even an O3 processor with many ALUs cannot overlap these —
 * it must wait for each result before issuing the next.
 *
 * Expected Konata view:
 *   Instructions queue up in the ROB but cannot be issued.
 *   Long gaps between issue and writeback of dependent instructions.
 *   IPC will be close to 1 (or even below due to front-end width=2).
 *
 */

#include <stdint.h>
#include <gem5/m5ops.h>


#define ITERATIONS 1000

volatile uint64_t sink;

int main(void) {
    uint64_t volatile val = 1;

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif

    for (int i = 0; i < ITERATIONS; i++) {
        /*
         * TRUE DEPENDENCY CHAIN: each operation consumes
         * the result of the previous one.
         * Latency of IntALU = 1 cycle (see CPU config),
         * so the chain throughput is 1 instruction/cycle regardless
         * of how wide the issue stage is.
         *
         * This is the "worst case" for an O3 processor —
         * it has all the hardware but cannot use it.
         */
        val = val + 1;    /* depends on previous val */
        val = val * 3;    /* depends on val above    (IntMult lat=3) */
        val = val + 7;    /* depends on val above    */
        val = val * 2;    /* depends on val above    (IntMult lat=3) */
        val = val - 4;    /* depends on val above    */
        val = val + val;  /* depends on val above    */
        val = val - 1;    /* depends on val above    */
        val = val * 5;    /* depends on val above    (IntMult lat=3) */
    }

    sink = val;

    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    return 0;
}
