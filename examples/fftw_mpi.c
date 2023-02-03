
// example of one-dimensional FFT using MPI

#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

double cali_cost = 0;
int call_count = 0;
double write_cost = 0;

int main(int argc, char* argv[]) {
    fftw_complex *x; // input array
    fftw_complex *y; // output array

    int nprocs, rank;
    int i, ret;
    fftw_plan forward_plan;
    ptrdiff_t N;
    ptrdiff_t size;
    ptrdiff_t local_ni, local_i_start, local_no, local_o_start;

    int RUNS = 20;
//    double start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc<2) {
        printf("Error give proper args!!");
        MPI_Finalize();
        return 0;
    }

    N = atoi(argv[1]);

    fftw_mpi_init(); // initial FFTW MPI Library

    double start_time = MPI_Wtime();

    double start = MPI_Wtime();
    cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");

	cali::ConfigManager mgr;
	mgr.add("hatchet-region-profile(output=sim_cali_512_ws,output.format=hatchet)");
    mgr.start();
	double end = MPI_Wtime();
	write_cost += (end - start);

    for(int run = 0; run < RUNS; run++) {

    	std::string name = "run_" + std::to_string(run);
    	start = MPI_Wtime();
    	CALI_MARK_BEGIN(name.data());
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("assignData");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        size = fftw_mpi_local_size_1d(N, MPI_COMM_WORLD, 
                FFTW_FORWARD, FFTW_ESTIMATE, &local_ni, 
                &local_i_start, &local_no, &local_o_start);

    	start = MPI_Wtime();
    	CALI_MARK_END("assignData");
    	end = MPI_Wtime();
    	cali_cost += (end - start);

    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("allocFFT");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        // fftw_malloc is similar to malloc except that it properly aligns the array when SIMD instructions
        // fftw_complex is a double[2] composed of the real and imaginary parts of a complex number.
        x = (fftw_complex *)fftw_malloc(size*sizeof(fftw_complex)); // assign input array
        y = (fftw_complex *)fftw_malloc(size*sizeof(fftw_complex)); // assign output array

    	start = MPI_Wtime();
    	CALI_MARK_END("allocFFT");
    	end = MPI_Wtime();
    	cali_cost += (end - start);


    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("initialFFT");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        for(int i = 0; i < size; i++) {
            x[i][0] = i;
            x[i][1] = i;
        }

    	start = MPI_Wtime();
    	CALI_MARK_END("initialFFT");
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        // create a plan: an object that contains all the data that FFTW needs to compute the FFT.
        // N: is the size of the transform you are trying to compute.
        // x and y: pointers to the input and output arrays. x equals to y indicate an in-place transform.
        // sign (FFTW_FORWARD (-1) or FFTW_BACKWARD (+1)) indicates the direction of the transform.
        // flag (FFTW_MEASURE or FFTW_ESTIMATE).
        // FFTW_MEASURE try to find the best way to do FFT with size n.
        // FFTW_ESTIMATE does not run any computation and just builds a reasonable plan.
        // The plan creation is a collective function that must be called for all processes in the communicator.
        // The in and out pointers refer only to a portion of the overall transform data as specified by the ‘local_size’ functions in the previous section.
        // Data distribution: e.g., 100 × 200 complex DFT and 4 processes, each process will get a 25 × 200 slice of the data (depends on the rows).
        // If the data cannot evenly distributed, the extra data will be assigned to process 0.
        // This distribution can also be specified by using several routines.


    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("plan");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        forward_plan = fftw_mpi_plan_dft_1d(N, x, y, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    	start = MPI_Wtime();
    	CALI_MARK_END("plan");
    	end = MPI_Wtime();
    	cali_cost += (end - start);

    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("execute");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        fftw_execute(forward_plan);

    	start = MPI_Wtime();
    	CALI_MARK_END("execute");
    	end = MPI_Wtime();
    	cali_cost += (end - start);


    	start = MPI_Wtime();
    	CALI_MARK_BEGIN("desyPlan");
    	call_count += 1;
    	end = MPI_Wtime();
    	cali_cost += (end - start);

        fftw_destroy_plan(forward_plan);

    	start = MPI_Wtime();
    	CALI_MARK_END("desyPlan");
    	end = MPI_Wtime();
    	cali_cost += (end - start);

    	start = MPI_Wtime();
    	CALI_MARK_END(name.data());
    	end = MPI_Wtime();
    	cali_cost += (end - start);
    }

	double end_time = MPI_Wtime();
	double time = (end_time-start_time);

	double swt = MPI_Wtime();
	mgr.flush();
    double ewt = MPI_Wtime();
    write_cost = (ewt - swt);

	double total_time = time + write_cost;
	double max_time;
	MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (total_time == max_time || rank == 0) { printf("Caliper-time(%d %d): %f, %f, %f, %d\n", nprocs, rank, time, cali_cost, write_cost, call_count); }


    fftw_mpi_cleanup(); // clean FFTW MPI Library
    MPI_Finalize();

    return 0;
}
