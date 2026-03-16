

#define N 100
#define ROUNDS 1
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>
// illustrates imbalanced parallelism
// in the calculation of cvec[i] = sum(mat[i][j] * bvec[j])



void fillMatrix(double* matrix, int size, double value) {
    for (int i = 0; i < size; i++)
        matrix[i] = value;
}

int main() {

    double *mat = (double*)malloc(N * N * sizeof(double));
    double *bvec = (double*)malloc(N * sizeof(double));
    double *cvec = (double*)malloc(N * sizeof(double));

    if (mat == NULL || bvec == NULL || cvec == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }   

    int i, j, k;
    // Initialization
    fillMatrix(mat, N*N, 1); 
    fillMatrix(bvec, N, 2);
    // Calculation loop
    omp_set_num_threads(omp_get_num_procs());

    printf("Number of threads: %d\n", omp_get_num_threads());
    
    #ifdef GEM5
	// m5_work_begin(work_id, thread_id) -- begin a item sample
    m5_work_begin(0, 0);
	#endif

    int current = 0;

    #pragma omp parallel
    {
        for ( k = 0; k < ROUNDS; k++) 
        {

        #pragma omp for private(j,current) schedule(static, 16)
            for ( i = 0; i < N; i++) {
                current = 0;
                for ( j = i; j < N; j++)
                    current += mat[(i*N)+j] * bvec[j];
                cvec[i] = current;
            }
        }
    }

    #ifdef GEM5
    // m5_work_end(work_id, thread_id) -- end a item sample
    m5_work_end(0, 0);
    #endif

    printf("Done\n");

    // Free allocated memory
    free(mat);
    free(bvec);
    free(cvec);
    return 0;
}
