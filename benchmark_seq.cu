/* include petsc */
#include "petsc.h"
#include "mpi.h"

#define PRINT_VECTOR_CONTENT 1

#ifdef USE_CUDA
	#include "include/this_is_kernel.h"
#else
	#include "include/this_will_be_kernel.h"
#endif


/* to deal with errors, call Petsc functions with TRY(fun); original idea from Permon (Vaclav Hapla) */
static PetscErrorCode ierr; 
#define TRY( f) {ierr = f; do {if (PetscUnlikely(ierr)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_IN_CXX,0);}} while(0);}

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

#ifdef USE_CUDA
	TRY( VecCreateSeqCUDA(PETSC_COMM_SELF, T_local*n, &x_local) );
#else
	TRY( VecCreateSeq(PETSC_COMM_SELF, T_local*n, &x_local) );
#endif

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

#ifdef USE_CUDA
	this_is_kernel<<<T_local, 1>>>(x_local_arr,T_local,n);
#else
	/* in this seq implementation "i" denotes the index of the kernel */
	for(i = 0; i < T_local; i++){
		this_will_be_kernel(i,x_local_arr,T_local,n);
	}
#endif

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


