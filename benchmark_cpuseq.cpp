/* include petsc */
#include "petsc.h"
#include "mpi.h"

#define PRINT_VECTOR_CONTENT 1

/* to deal with errors, call Petsc functions with TRY(fun); original idea from Permon (Vaclav Hapla) */
static PetscErrorCode ierr; 
#define TRY( f) {ierr = f; do {if (PetscUnlikely(ierr)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_IN_CXX,0);}} while(0);}

void this_will_be_kernel(int t, double *x, int T, int K); /* see implementation after main function */
void device_sort_bubble(double *x, int n);

int main( int argc, char *argv[] )
{
	/* problem dimensions */
	int T = 11;
	int n = 3;
	
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

	TRY( PetscPrintf(PETSC_COMM_WORLD,"This is Petsc-VECSEQ sample.\n") );
	TRY( PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n") );
	
/* SAY HELLO TO WORLD - to check if everything is working */
	
	/* give info about MPI */
	int size, rank; /* size and rank of communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	TRY( PetscPrintf(MPI_COMM_WORLD,"- number of processors: %d\n",size) );
	TRY( PetscSynchronizedPrintf(MPI_COMM_WORLD," - hello from processor: %d\n",rank) );
	TRY( PetscSynchronizedFlush(MPI_COMM_WORLD,NULL) );
	TRY( PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n") );

/* CREATE GLOBAL VECTOR */

	/* create layout vector to figure out how much this proc will compute */
	Vec layout; /* auxiliary vector */
	TRY( VecCreate(PETSC_COMM_WORLD,&layout) );
	TRY( VecSetSizes(layout,PETSC_DECIDE,T) );
	TRY( VecSetFromOptions(layout) );

	int T_local; /* local portion of "time-series" */
	TRY( VecGetLocalSize(layout,&T_local) );
	TRY( VecDestroy(&layout) ); /* destroy testing vector - it is useless now */

	/* print info about sizes */
	TRY( PetscPrintf(MPI_COMM_WORLD,"- global T: %d\n",T) );
	TRY( PetscSynchronizedPrintf(MPI_COMM_WORLD," [%d]: local T: %d\n",rank,T_local) );
	TRY( PetscSynchronizedFlush(MPI_COMM_WORLD,NULL) );
	TRY( PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n") );


	/* now create the data vector */
	Vec x_global; /* global data vector */
	TRY( VecCreate(PETSC_COMM_WORLD,&x_global) );
	TRY( VecSetSizes(x_global,T_local*n,PETSC_DECIDE) );
	TRY( VecSetFromOptions(x_global) );

	/* set some random values - just for fun */
	PetscRandom rnd; /* random generator */
	TRY( PetscRandomCreate(PETSC_COMM_WORLD,&rnd) );
	TRY( PetscRandomSetType(rnd,PETSCRAND) );
	TRY( PetscRandomSetFromOptions(rnd) );
	TRY( PetscRandomSetSeed(rnd,13) );

	/* generate random data */
	TRY( VecSetRandom(x_global, rnd) );

	/* destroy the random generator */
	TRY( PetscRandomDestroy(&rnd) );

	/* maybe print the content of the global vector ? */
	if(PRINT_VECTOR_CONTENT){
		TRY( VecView(x_global, PETSC_VIEWER_STDOUT_WORLD) );
	}

/* GET LOCAL VECTOR - SEQGPU operations will be performed only on this vector, 
 * operations are completely independent, there is not GPU-GPU communication at all,
 * after the computation, we will return local vector back to global */
 
	/* prepare local vector */
	Vec x_local; /* local data vector */
	TRY( VecCreateSeq(PETSC_COMM_SELF, T_local*n, &x_local) ); /* TODO: change this to VECSEQCUDA */

	/* get local vector */
	TRY( VecGetLocalVector(x_global,x_local) ); /* actually, this is quite new Petsc feature, the reason why I installed newer version on my PC */


/* -------------------------------------------------
 * PERFORM SOME OPERATIONS ON LOCAL VECTOR 
 * this is the part where GPU operations on x_local will be implemented 
 * -------------------------------------------------
*/
	
	/* fun with subvectors - get subvectors x_local = [xsub0, xsub1, ..., xsub{n-1}] */
	IS xsub_is[n];
	Vec xsub[n];
	int i;
	for(i=0;i<n;i++){
		TRY( ISCreateStride(PETSC_COMM_SELF, T_local, i*T_local, 1, &xsub_is[i]) );
		TRY( VecGetSubVector(x_local, xsub_is[i], &xsub[i]) );
	}

	/* compute some BLAS operations on subvectors */
	double result;
	
	/* NORM_2 */
	TRY( VecNorm(xsub[0], NORM_2, &result) );
	TRY( PetscPrintf(MPI_COMM_WORLD,"- test norm: %f\n",result) );

	/* Duplicate */
	Vec y;
	TRY( VecDuplicate(xsub[0],&y) );
	
	/* VecSet */
	TRY( VecSet(y, 1.0) );
	
	/* MAXPY (multiple AXPY) - for our matrix-vector free operations */
	double coeff[n];
	for(i=0;i<n;i++){
		coeff[i]= 1/(double)(i+1);
	}
	TRY( VecMAXPY(y, n, coeff, xsub) );

	/* dot product */
	TRY( VecDot(y,xsub[0], &result) );
	TRY( PetscPrintf(MPI_COMM_WORLD,"- test dot: %f\n",result) );

	/* scale */
	TRY( VecScale(y, -result) );

	/* pointwisemult */	
	for(i=0;i<n;i++){
		TRY( VecPointwiseMult(xsub[i], xsub[i], y) );
	}

	/* destroy temp vector */
	TRY( VecDestroy(&y) );

	/* VecSum */
	for(i=0;i<n;i++){
		TRY( VecSum(xsub[i], &result) );
		TRY( VecScale(xsub[i], 1.0/(double)result) );
	}

	/* restore subvectors */
	for(i=0;i<n;i++){
		TRY( VecRestoreSubVector(x_local, xsub_is[i], &xsub[i]) );
		TRY( ISDestroy(&xsub_is[i]) );
	}


/* KERNEL call */

	/* get local array (for own kernels) */
	double *x_local_arr;
	TRY( VecGetArray(x_local,&x_local_arr) );

	/* todo: call kernel (number of kernels is Tlocal) */
	// this_is_kernel<<<Tlocal, 1>>>(x_local_arr,Tlocal,n);
	
	/* in this seq implementation "i" denotes the index of the kernel */
	for(i = 0; i < T_local; i++){
		this_will_be_kernel(i,x_local_arr,T_local,n);
	}

	/* restore local array */
	TRY( VecRestoreArray(x_local,&x_local_arr) );


/* -------------------------------------------------
 * end of CUDA-suitable operations 
 * -------------------------------------------------
*/


/* LOCAL BACK TO GLOBAL */

	/* restore local vector back to global */
	TRY( VecRestoreLocalVector(x_global,x_local) );
	
	/* maybe print the content of the global vector ? */
	if(PRINT_VECTOR_CONTENT){
		TRY( VecView(x_global, PETSC_VIEWER_STDOUT_WORLD) );
	}


	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


/* this will be a kernel function which computes a projection of a point onto simplex in nD
 *
 * just for curiosity more details: 
 * take K-dimensional vector x[t,t+T,t+2T,...t+(K-1)T] =: p
 * and compute projection
 * P(p) = arg min || p - y ||_2
 * subject to constraints (which define simplex)
 * y_0 + ... y_{K-1} = 1
 * y_i >= 0 for all i=0,...,K-1
 *
 * in practical applications K is much more lower number than T
 * K - number of clusters (2 - 10^2)
 * T - length of time-series (10^5 - 10^9) 
 */ 
void this_will_be_kernel(int t, double *x, int T, int K){
//__global__ void this_is_kernel(double *x, int T, int K){
//	int t = blockIdx.x*blockDim.x + threadIdx.x; /* compute my id */

	if(t<T){ /* maybe we call more than T kernels */
		int k;

		bool is_inside = true;
		double sum = 0.0;
	
		/* control inequality constraints */
		for(k = 0; k < K; k++){ // TODO: could be performed parallely  
			if(x[k*T+t] < 0.0){
				is_inside = false;
			}
			sum += x[k*T+t];
		}

		/* control equality constraints */
		if(sum != 1){ 
			is_inside = false;
		}

		/* if given point is not inside the feasible domain, then do projection */
		if(!is_inside){
			int j,i;
			/* compute sorted x_sub */
			double *y = new double[K];
			double sum_y;
			for(k=0;k<K;k++){
				y[k] = x[k*T+t]; 
			}
			device_sort_bubble(y,K);

			/* now perform analytical solution of projection problem */	
			double t_hat = 0.0;
			i = K - 1;
			double ti;

			while(i >= 1){
				/* compute sum(y) */
				sum_y = 0.0;
				for(j=i;j<K;j++){ /* sum(y(i,n-1)) */
					sum_y += y[j];
				}
				
				ti = (sum_y - 1.0)/(double)(K-i);
				if(ti >= y[i-1]){
					t_hat = ti;
					i = -1; /* break */
				} else {
					i = i - 1;
				}
			}

			if(i == 0){
				t_hat = (sum-1.0)/(double)K; /* uses sum=sum(x_sub) */
			}
    
			for(k = 0; k < K; k++){ // TODO: could be performed parallely  
				/* (*x_sub)(i) = max(*x_sub-t_hat,0); */
				ti = x[k*T+t] - t_hat;	
				if(ti > 0.0){
					x[k*T+t] = ti;
				} else {
					x[k*T+t] = 0.0;
				}
			}
			
			delete y;
		}
		
	}

	/* if t >= T then relax and do nothing */	
}


//__device__ void device_sort_bubble(double *x, int n);
void device_sort_bubble(double *x, int n){
	int i;
	int m = n;
	int mnew;
	double swap;

	while(m > 0){
		/* Iterate through x */
		mnew = 0;
		for(i=1;i<m;i++){
			/* Swap elements in wrong order */
			if (x[i] < x[i - 1]){
				swap = x[i];
				x[i] = x[i-1];
				x[i-1] = swap;
				mnew = i;
			}
        }
		m = mnew;
    }
}
