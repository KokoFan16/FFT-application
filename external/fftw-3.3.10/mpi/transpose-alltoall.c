/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* plans for distributed out-of-place transpose using MPI_Alltoall,
   and which destroy the input array (unless TRANSPOSED_IN is used) */

#include "mpi-transpose.h"
#include <string.h>

int count = 0;

typedef struct {
     solver super;
     int copy_transposed_in; /* whether to copy the input for TRANSPOSED_IN,
				which makes the final transpose out-of-place
				but costs an extra copy and requires us
				to destroy the input */
} S;

typedef struct {
     plan_mpi_transpose super;

     plan *cld1, *cld2, *cld2rest, *cld3;

     MPI_Comm comm;
     int *send_block_sizes, *send_block_offsets;
     int *recv_block_sizes, *recv_block_offsets;

     INT rest_Ioff, rest_Ooff;

     int equal_blocks;
} P;

static void apply(const plan *ego_, R *I, R *O)
{
	 int rank, nprocs;
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//	 if (rank == 0)
//		 printf("%d: %s\n", rank, "transpose-alltoall");

     const P *ego = (const P *) ego_;
     plan_rdft *cld1, *cld2, *cld2rest, *cld3;

     /* transpose locally to get contiguous chunks */
     cld1 = (plan_rdft *) ego->cld1;
     if (cld1) {
	  cld1->apply(ego->cld1, I, O);
	  
	  /* transpose chunks globally */
	  if (ego->equal_blocks)
	  {
		 if (rank == 0)
			 printf("%d: %s\n", rank, "1--MPI_Alltoall");
	       MPI_Alltoall(O, ego->send_block_sizes[0], FFTW_MPI_TYPE,
			    I, ego->recv_block_sizes[0], FFTW_MPI_TYPE,
			    ego->comm);
	  }
	  else {

			 if (rank == 0)
				 printf("%d: %s\n", rank, "2--MPI_Alltoallv");

	       MPI_Alltoallv(O, ego->send_block_sizes, ego->send_block_offsets,
			     FFTW_MPI_TYPE,
			     I, ego->recv_block_sizes, ego->recv_block_offsets,
			     FFTW_MPI_TYPE,
			     ego->comm);
	  }
     }
     else { /* TRANSPOSED_IN, no need to destroy input */
	  /* transpose chunks globally */
	  if (ego->equal_blocks){

		   double start = MPI_Wtime();
		   uniform_modified_inverse_r_bruck((char*)I, ego->send_block_sizes[0], FFTW_MPI_TYPE,
					 (char*)O, ego->recv_block_sizes[0], FFTW_MPI_TYPE, ego->comm);
//	       MPI_Alltoall(I, ego->send_block_sizes[0], FFTW_MPI_TYPE,
//			    O, ego->recv_block_sizes[0], FFTW_MPI_TYPE,
//			    ego->comm);
	       double end = MPI_Wtime();
	       if (rank == 0)
	    	   printf("My-MPI_Alltoall-%d %d %f\n", count, nprocs, (end - start));

	       count += 1;
	  }
	  else {
//		  	  if (rank == 0)
//				 printf("%d: %s\n", rank, "4--MPI_Alltoallv");

	       MPI_Alltoallv(I, ego->send_block_sizes, ego->send_block_offsets,
			     FFTW_MPI_TYPE,
			     O, ego->recv_block_sizes, ego->recv_block_offsets,
			     FFTW_MPI_TYPE,
			     ego->comm);
	  }
	  I = O; /* final transpose (if any) is in-place */
     }
     
     /* transpose locally, again, to get ordinary row-major */
     cld2 = (plan_rdft *) ego->cld2;
     if (cld2) {
	  cld2->apply(ego->cld2, I, O);
	  cld2rest = (plan_rdft *) ego->cld2rest;
	  if (cld2rest) { /* leftover from unequal block sizes */
	       cld2rest->apply(ego->cld2rest,
			       I + ego->rest_Ioff, O + ego->rest_Ooff);
	       cld3 = (plan_rdft *) ego->cld3;
	       if (cld3)
		    cld3->apply(ego->cld3, O, O);
	       /* else TRANSPOSED_OUT is true and user wants O transposed */
	  }
     }
}

static int applicable(const S *ego, const problem *p_,
		      const planner *plnr)
{
     const problem_mpi_transpose *p = (const problem_mpi_transpose *) p_;
     return (1
	     && p->I != p->O
	     && (!NO_DESTROY_INPUTP(plnr) || 
		 ((p->flags & TRANSPOSED_IN) && !ego->copy_transposed_in))
	     && ((p->flags & TRANSPOSED_IN) || !ego->copy_transposed_in)
	     && ONLY_TRANSPOSEDP(p->flags)
	  );
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;
     X(plan_awake)(ego->cld1, wakefulness);
     X(plan_awake)(ego->cld2, wakefulness);
     X(plan_awake)(ego->cld2rest, wakefulness);
     X(plan_awake)(ego->cld3, wakefulness);
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     X(ifree0)(ego->send_block_sizes);
     MPI_Comm_free(&ego->comm);
     X(plan_destroy_internal)(ego->cld3);
     X(plan_destroy_internal)(ego->cld2rest);
     X(plan_destroy_internal)(ego->cld2);
     X(plan_destroy_internal)(ego->cld1);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     p->print(p, "(mpi-transpose-alltoall%s%(%p%)%(%p%)%(%p%)%(%p%))",
	      ego->equal_blocks ? "/e" : "",
	      ego->cld1, ego->cld2, ego->cld2rest, ego->cld3);
}

static plan *mkplan(const solver *ego_, const problem *p_, planner *plnr)
{
     const S *ego = (const S *) ego_;
     const problem_mpi_transpose *p;
     P *pln;
     plan *cld1 = 0, *cld2 = 0, *cld2rest = 0, *cld3 = 0;
     INT b, bt, vn, rest_Ioff, rest_Ooff;
     R *I;
     int *sbs, *sbo, *rbs, *rbo;
     int pe, my_pe, n_pes;
     int equal_blocks = 1;
     static const plan_adt padt = {
          XM(transpose_solve), awake, print, destroy
     };

     if (!applicable(ego, p_, plnr))
          return (plan *) 0;

     p = (const problem_mpi_transpose *) p_;
     vn = p->vn;

     MPI_Comm_rank(p->comm, &my_pe);
     MPI_Comm_size(p->comm, &n_pes);

     b = XM(block)(p->nx, p->block, my_pe);

     if (p->flags & TRANSPOSED_IN) { /* I is already transposed */
	  if (ego->copy_transposed_in) {
	       cld1 = X(mkplan_f_d)(plnr,
				  X(mkproblem_rdft_0_d)(X(mktensor_1d)
							(b * p->ny * vn, 1, 1),
							I = p->I, p->O),
				    0, 0, NO_SLOW);
	       if (XM(any_true)(!cld1, p->comm)) goto nada;
	  }
	  else
	       I = p->O; /* final transpose is in-place */
     }
     else { /* transpose b x ny x vn -> ny x b x vn */
	  cld1 = X(mkplan_f_d)(plnr, 
			       X(mkproblem_rdft_0_d)(X(mktensor_3d)
						     (b, p->ny * vn, vn,
						      p->ny, vn, b * vn,
						      vn, 1, 1),
						     I = p->I, p->O),
			       0, 0, NO_SLOW);
	  if (XM(any_true)(!cld1, p->comm)) goto nada;
     }
	  
     if (XM(any_true)(!XM(mkplans_posttranspose)(p, plnr, I, p->O, my_pe,
						 &cld2, &cld2rest, &cld3,
						 &rest_Ioff, &rest_Ooff),
		      p->comm)) goto nada;

     pln = MKPLAN_MPI_TRANSPOSE(P, &padt, apply);

     pln->cld1 = cld1;
     pln->cld2 = cld2;
     pln->cld2rest = cld2rest;
     pln->rest_Ioff = rest_Ioff;
     pln->rest_Ooff = rest_Ooff;
     pln->cld3 = cld3;

     MPI_Comm_dup(p->comm, &pln->comm);

     /* Compute sizes/offsets of blocks to send for all-to-all command. */
     sbs = (int *) MALLOC(4 * n_pes * sizeof(int), PLANS);
     sbo = sbs + n_pes;
     rbs = sbo + n_pes;
     rbo = rbs + n_pes;
     b = XM(block)(p->nx, p->block, my_pe);
     bt = XM(block)(p->ny, p->tblock, my_pe);
     for (pe = 0; pe < n_pes; ++pe) {
	  INT db, dbt; /* destination block sizes */
	  db = XM(block)(p->nx, p->block, pe);
	  dbt = XM(block)(p->ny, p->tblock, pe);
	  if (db != p->block || dbt != p->tblock)
	       equal_blocks = 0;

	  /* MPI requires type "int" here; apparently it
	     has no 64-bit API?  Grrr. */
	  sbs[pe] = (int) (b * dbt * vn);
	  sbo[pe] = (int) (pe * (b * p->tblock) * vn);
	  rbs[pe] = (int) (db * bt * vn);
	  rbo[pe] = (int) (pe * (p->block * bt) * vn);
     }
     pln->send_block_sizes = sbs;
     pln->send_block_offsets = sbo;
     pln->recv_block_sizes = rbs;
     pln->recv_block_offsets = rbo;
     pln->equal_blocks = equal_blocks;

     X(ops_zero)(&pln->super.super.ops);
     if (cld1) X(ops_add2)(&cld1->ops, &pln->super.super.ops);
     if (cld2) X(ops_add2)(&cld2->ops, &pln->super.super.ops);
     if (cld2rest) X(ops_add2)(&cld2rest->ops, &pln->super.super.ops);
     if (cld3) X(ops_add2)(&cld3->ops, &pln->super.super.ops);
     /* FIXME: should MPI operations be counted in "other" somehow? */

     return &(pln->super.super);

 nada:
     X(plan_destroy_internal)(cld3);
     X(plan_destroy_internal)(cld2rest);
     X(plan_destroy_internal)(cld2);
     X(plan_destroy_internal)(cld1);
     return (plan *) 0;
}

static solver *mksolver(int copy_transposed_in)
{
     static const solver_adt sadt = { PROBLEM_MPI_TRANSPOSE, mkplan, 0 };
     S *slv = MKSOLVER(S, &sadt);
     slv->copy_transposed_in = copy_transposed_in;
     return &(slv->super);
}

void XM(transpose_alltoall_register)(planner *p)
{
     int cti;
     for (cti = 0; cti <= 1; ++cti)
	  REGISTER_SOLVER(p, mksolver(cti));
}


static int myPow(int x, unsigned int p) {
	if (p == 0) return 1;
	if (p == 1) return x;

	int tmp = myPow(x, p/2);
	if (p%2 == 0) return tmp * tmp;
	else return x * tmp * tmp;
}

/// all-to-all imp with r = sqrt(P)
void uniform_modified_inverse_r_bruck(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm) {

	int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int r = sqrt(nprocs);

    int unit_size = sendcount * typesize;
    int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	int nlpow = pow(r, w-1);
	int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	char* temp_send;
	if (sendbuf == MPI_IN_PLACE) {
		temp_send = (char*)malloc(nprocs * unit_size); // temporary buffer
		memcpy(temp_send, recvbuf, nprocs * unit_size);
	}
	else {
		temp_send = sendbuf;
	}


	for (int i = 0; i < nprocs; i++) {
		int index = (2*rank-i+nprocs)%nprocs;
		memcpy(recvbuf+(index*unit_size), temp_send+(i*unit_size), unit_size);
	}

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	int comm_steps = (r - 1)*w - d;
	char* temp_buffer = (char*)malloc(nlpow * unit_size); // temporary buffer
	int spoint = 1, distance = myPow(r, w-1), next_distance = distance*r;
    for (int x = w-1; x > -1; x--) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = ze-1; z > 0; z--) {
    		// get the sent data-blocks
    		// copy blocks which need to be sent at this step
    		di = 0; ci = 0;
			spoint = z * distance;
			for (int i = spoint; i < nprocs; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > nprocs - 1 ) { break; }
					int id = (j + rank) % nprocs;
					sent_blocks[di++] = id;
					memcpy(&temp_buffer[unit_size*ci++], &recvbuf[id*unit_size], unit_size);
				}
			}

    		// send and receive
    		int recv_proc = (rank + spoint) % nprocs; // receive data from rank - 2^step process
    		int send_proc = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process
    		long long comm_size = di * unit_size;
    		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, temp_send, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

    		// replace with received data
    		for (int i = 0; i < di; i++) {
    			long long offset = sent_blocks[i] * unit_size;
    			memcpy(recvbuf+offset, temp_send+(i*unit_size), unit_size);
    		}
    	}
		distance /= r;
		next_distance /= r;
    }
	free(temp_buffer);
	free(temp_send);
}
