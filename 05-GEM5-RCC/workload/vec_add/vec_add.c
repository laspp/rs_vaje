#include <stdlib.h>   //malloc and free
#include <stdio.h>    //printf
#include <omp.h>      //OpenMP
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>



// Very small values for this simple illustrative example
#define ARRAY_SIZE 100000    //Size of arrays whose elements will be added together.


int main (int argc, char *argv[]) 
{
	// elements of arrays a and b will be added
	// and placed in array c
	int * a;
	int * b; 
	int * c;
        
    int n = ARRAY_SIZE;                 // number of array elements
	int n_per_thread;                   // elements per thread
    int total_threads = omp_get_max_threads();    // number of threads to use  
	int i;       // loop index
        
        // allocate spce for the arrays
    a = (int *) malloc(sizeof(int)*n);
	b = (int *) malloc(sizeof(int)*n);
	c = (int *) malloc(sizeof(int)*n);
	printf("Vector addition using %d threads for %d elements\n", total_threads, n);

    // initialize arrays a and b with consecutive integer values
    // as a simple example
	for(i=0; i<n; i++) {
		a[i] = i;
	}
	for(i=0; i<n; i++) {
		b[i] = i;
	}   
	

	omp_set_num_threads(total_threads);
	
	n_per_thread = n/total_threads;
	
   
	double start_time, end_time;
	start_time = omp_get_wtime();
	
	#ifdef GEM5
	m5_work_begin(0, 0);
	#endif
	#pragma omp parallel shared(a, b, c) private(i)
	{
		
		#pragma omp for schedule(static, n_per_thread)
		for(i=0; i<n; i++) {
			c[i] = a[i]+b[i];
		}
	}
	#ifdef GEM5
	// Use the m5op_addr to input the "magic" address
	m5_work_end(0, 0);
	#endif
	end_time = omp_get_wtime();
	printf("Vector addition using %d threads for %d elements\n", total_threads, n);
	printf("Time is %f\n", end_time - start_time);

	
	// clean up memory
	free(a);  free(b); free(c);
	
	return 0;
}