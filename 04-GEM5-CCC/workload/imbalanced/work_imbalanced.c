#include <omp.h>
#include <stdio.h>
#include <math.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>
#define N 256

double result[N];

int main() {


    int total_threads = omp_get_max_threads();    // number of threads to use  
    printf("Running with %d threads\n", total_threads);
    #ifdef GEM5
    // m5_work_begin(work_id, thread_id) -- begin a item sample
    m5_work_begin(0, 0);
    #endif

    #pragma omp parallel for schedule(dynamic,16) 
    for (int i = 0; i < N; i++) {
        /* Iteration i does O(i) work — later iterations are much heavier */
        double val = 0.0;
        for (int j = 0; j < i*10; j++) {
            val += sqrt((double)(i * j + 1));
        }
        result[i] = val;
    }
    #ifdef GEM5
    // m5_work_end(work_id, thread_id) -- end the item sample
    m5_work_end(0, 0);
    #endif
    return 0;
}