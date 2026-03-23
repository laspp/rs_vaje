
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>

/******************************************************************************/

int main (int argc, char *argv[]) {
    double a[100][100];
    double b[100][100];
    double c[100][100];
    int i, j, k, n = 100;;

    int thread_num;


    thread_num = omp_get_max_threads ( );


    printf ( "\n" );
    printf ( "  The number of processors available = %d\n", omp_get_num_procs ( ) );
    printf ( "  The number of threads available    = %d\n", thread_num );
    printf ( "  The matrix order N                 = %d\n", n );

    for ( i = 0; i < n; i++ ) {
        for ( j = 0; j < n; j++ ) {
            a[i][j] = 1.0;
            b[i][j] = a[i][j];
        }
    }




#ifdef GEM5
	m5_work_begin(0, 0);
#endif

# pragma omp parallel shared ( a, b, c, n ) private ( i, j, k )
{
    # pragma omp for
    for ( i = 0; i < n; i++ ){
        for ( j = 0; j < n; j++ ){
            c[i][j] = 0.0;
            for ( k = 0; k < n; k++ ){
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}
#ifdef GEM5
	// Use the m5op_addr to input the "magic" address
	m5_work_end(0, 0);
#endif  
    return 0;
}
