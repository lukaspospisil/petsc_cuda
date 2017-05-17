/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* for measuring computation time */
#include "include/timer.h"
#ifdef USE_CUDA
	#include "include/cuda_stuff.h"
#endif

/* parameters of the problem */
#define X_SIZE 1000000
#define QUOTIENT 0.5
#define N_TRIALS 1000
#define PRINT_VECTOR_CONTENT 0

/* set values of the vector */
void set_values(Vec &x, double q, int low, int n_local);
void set_values_cuda(Vec &x, double q, int low, int n_local);

int main( int argc, char *argv[] )
{
	/* error handling */
	PetscErrorCode ierr; 
	
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

/* SAY HELLO TO WORLD - to check if everything is working */
	
	/* give info about MPI */
	int size, rank; /* size and rank of communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- Hello World: ---------------\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(MPI_COMM_WORLD,"- number of processors: %d\n",size); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD," - hello from processor: %d\n",rank); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(MPI_COMM_WORLD,NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);

/* CREATE GLOBAL VECTOR */
	/* problem parameters */
	int n = X_SIZE; /* (global) length of vector */
	double q = QUOTIENT; /* quotient of the geometrical sequence */
	double theory_sum = 1.0/(1-q); /* theoretical sum of the geometrical sequence */

	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- Problem info: ----------------\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- n = %d\n",n); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- q = %f\n",q); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- n_trials = %d\n",N_TRIALS); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);

	/* create layout vector to figure out how much this proc will compute */
	Vec x; /* auxiliary vector */
	ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n); CHKERRQ(ierr);
	/* set the type of vector */
	#ifdef USE_CUDA
		ierr = VecSetType(x, VECMPICUDA); CHKERRQ(ierr);
	#else
		ierr = VecSetType(x, VECMPI); CHKERRQ(ierr);
	#endif
	ierr = VecSetFromOptions(x); CHKERRQ(ierr);

	/* local portion of vector */
	int n_local; 
	ierr = VecGetLocalSize(x,&n_local); CHKERRQ(ierr);

	/* ownership range */
	int low, high;
	ierr = VecGetOwnershipRange(x, &low, &high); CHKERRQ(ierr);

	/* print info about sizes */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- Vector info: ----------------\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- global size: %d\n",n); CHKERRQ(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_SELF," [%d] local size: %d, low: %d, high: %d\n",rank,n_local,low,high); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);

	/* set values of vector x[i] = q^i, i=1,\dots,n */
	#ifdef USE_CUDA
		set_values_cuda(x,q,low,n_local);
	#else
		set_values(x,q,low,n_local);
	#endif

	/* maybe print the content of the global vector ? */
	if(PRINT_VECTOR_CONTENT){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- Vector content: -------------\n"); CHKERRQ(ierr);
		ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	}

/* COMPUTE SUM OF THE VECTOR */
	Timer mytimer; /* I am measuring time using this thing */
	mytimer.restart();

	double mysum = -1.0; /* here will be stored the value of sum */
	
	mytimer.start();
	for(int i=0;i<N_TRIALS;i++){
		ierr = VecSum(x,&mysum); CHKERRQ(ierr);
	}
	mytimer.stop();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- SUM info: ----------------\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- theoretical sum       : %f\n",theory_sum); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- computed sum          : %f\n",mysum); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- difference            : %g\n",std::abs(mysum-theory_sum)); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total time            : %g s\n", mytimer.get_value_last()); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  time            : %g s\n", mytimer.get_value_last()/(double)N_TRIALS); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);

	/* destroy vector */
	ierr = VecDestroy(&x); CHKERRQ(ierr);

	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


/* CPU variant of setting values */
void set_values(Vec &x, double q, int low, int n_local){
	PetscErrorCode ierr; 

	double *x_arr;
	ierr = VecGetArray(x, &x_arr); CHKERRV(ierr);
	for(int i=0;i<n_local;i++){
		x_arr[i] = pow(q,low+i); /* x[i] = q^i */
	}
	ierr = VecRestoreArray(x, &x_arr); CHKERRV(ierr);

}

/* CUDA variant of setting values */
#ifdef USE_CUDA
/* without cuda I am not able to even compile following code */

__global__ void set_values_kernel(double *x_arr, double q, int low, int n_local){
	int i = blockIdx.x*blockDim.x + threadIdx.x; /* thread index */

	if(i<n_local){
		x_arr[i] = pow(q,low+i); /* x[i] = q^i */
	} else {
		/* relax and take some beverage */
	}

}

void set_values_cuda(Vec &x, double q, int low, int n_local){
	PetscErrorCode ierr; 

	/* compute optimal cuda kernel call */
	int blockSize; /* block size returned by the launch configurator */
	int minGridSize; /* the minimum grid size needed to achieve the maximum occupancy for a full device launch */
	int gridSize; /* the actual grid size needed, based on input size */
	gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, set_values_kernel, 0, 0) );
	gridSize = (n_local + blockSize - 1)/ blockSize;

	double *x_arr;
	ierr = VecCUDAGetArrayReadWrite(x,&x_arr); CHKERRV(ierr);
	set_values_kernel<<<gridSize, blockSize>>>(x_arr,q,low,n_local);
	gpuErrchk( cudaDeviceSynchronize() );
	ierr = PetscBarrier(NULL); CHKERRV(ierr); /* includes cuda barrier (i.e. cudaDeviceSynchronize()) ? */
	ierr = VecCUDARestoreArrayReadWrite(x,&x_arr); CHKERRV(ierr);

}

#endif


