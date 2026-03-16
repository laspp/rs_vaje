


#define N 100
#define ROUNDS 2
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>

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
    
    

    // Initialization
    fillMatrix(mat, N*N, 1); 
    fillMatrix(bvec, N, 2);
    // Calculation loop
    omp_set_num_threads(omp_get_num_procs());

    printf("Number of threads: %d\n", omp_get_num_threads());
    


    int i, j, k;
    int current = 0;


    #ifdef GEM5
	// m5_work_begin(work_id, thread_id) -- begin a item sample
    m5_work_begin(0, 0);
	#endif

    #pragma omp parallel
    {
    // do the product of mat and bvec in parallel

    #pragma omp for private(j,current)
    for(i=0;i<N;i++) {
        current = 0;
        for ( j = 0; j < N; j++) 
            current += mat[(i*N)+j] * bvec[j];
        cvec[i] = current;
    }

    }

    #ifdef GEM5
	m5_work_end(0, 0);
	#endif



    // Free allocated memory
    free(mat);
    free(bvec);
    free(cvec);

    return 0;
}

