#include <stdlib.h>
#include <fftw3-mpi.h>


int main( int argc, char **argv)
{
     const ptrdiff_t N0 = 1024, N1 = 1024;
     fftw_plan plan;
     fftw_complex *in, *out;
     ptrdiff_t alloc_local, local_n0, local_0_start, i, j;
     double start, end;
     int rank, nprocs;

     MPI_Init( &argc, &argv );
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
     fftw_mpi_init();

     /* get local data size and allocate */
     alloc_local = fftw_mpi_local_size_2d( N0, N1, MPI_COMM_WORLD, &local_n0, &local_0_start );

     in = fftw_alloc_complex( alloc_local );
     out = fftw_alloc_complex( alloc_local );

     /* create plan for out-for-place forward DFT */
     plan = fftw_mpi_plan_dft_2d(N0, N1, in, out, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

     /* initialize data to some function my_function(x,y) */
     for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j)
     {
	    in[i*N1 + j][0] = ( double ) rand ( ) / ( double ) RAND_MAX;
        in[i*N1 + j][1] = ( double ) rand ( ) / ( double ) RAND_MAX;
     }

     start = MPI_Wtime();
     /* compute transforms, out-for-place, as many times as desired */
     fftw_execute(plan);
     end = MPI_Wtime();

     if(rank == 0)
    	 printf("%td %td %d %f\n", N0, N1, nprocs, end-start);
//
//     if (rank == 0) {
//		 for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j){
//			printf("%f\n", out[i*N1 + j][0]);
//		 }
//     }

     fftw_destroy_plan(plan);

     MPI_Finalize();

     return 0;
  
}
