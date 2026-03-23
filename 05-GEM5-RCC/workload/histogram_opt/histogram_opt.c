#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>

#define N1 100 // Number of rows in the image
#define N2 100 // Number of columns in the image
#define NUM_BINS 256 // Number of bins in the histogram

int main() {
    int img[N1][N2];
    long int histogram[NUM_BINS];
    int i, j;

    // Seed the random number generator
    srand(time(NULL));

    // Initialize the image with random values between 0 and NUM_BINS - 1
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            img[i][j] = rand() % NUM_BINS;
        }
    }

    // Initialize the histogram array
    
    // Get the number of threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    printf("Number of threads: %d\n", num_threads);
    // Parallelized histogram computation

    #ifdef GEM5
        m5_work_begin(0, 0);
    #endif


    int local_histogram[num_threads + 1][1024]; //for false sharing 
	memset(local_histogram, 0, sizeof(local_histogram));

    memset(histogram, 0, sizeof(histogram));
    #pragma omp parallel
    {
   
        int id = omp_get_thread_num();

        #pragma omp for private(j) 
        for (i = 0; i < N1; i++) {
            for (j = 0; j < N2; j++) {
                local_histogram[id][img[i][j]]++;
            }
        }
        

        #pragma omp critical
        {
            for (i = 0; i < NUM_BINS; i++) {
                histogram[i] += local_histogram[id][i];
            }
        }
    }

    #ifdef GEM5
        // Use the m5op_addr to input the "magic" address
        m5_work_end(0, 0);
    #endif  
    // Print the histogram
    printf("Histogram:\n");
    for (i = 0; i < NUM_BINS; i++) {
        printf("Value %d: %ld\n", i, histogram[i]);
    }

    return 0;
}
