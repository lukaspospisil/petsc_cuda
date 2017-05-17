/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* for measuring computation time */
#include "include/timer.h"
#ifdef USE_CUDA
	/* some cuda helpers */
	#include "include/cuda_stuff.h"
#endif

#define X_SIZE 1e6
#define N_TRIALS 1000
#define PRINT_VECTOR_CONTENT 0

int main( int argc, char *argv[] )
{
	/* error handling */
	PetscErrorCode ierr; 
	
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

	/* warm up GPU - call empty kernel (see include/cuda_stuff.h) */
	#ifdef USE_CUDA
		warm_up_cuda();
	#endif

	/* problem dimensions */
	int n = X_SIZE; /* length of vectors */
	int ntrials = N_TRIALS; /* number of trials (to provide average time) */
	
	/* print info about benchmark */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"This is IS test.\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - n          : %d\t\t(length of vectors)\n",n); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - ntrials    : %d\t\t(number of trials)\n",ntrials); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	
	Timer mytimer;
	mytimer.restart();
	mytimer.start();

	/* create vector x */
	Vec x;
	ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n); CHKERRQ(ierr);
	#ifdef USE_CUDA
		/* if we are using CUDA, it is a good idea to compute on GPU */
		ierr = VecSetType(x, VECMPICUDA); CHKERRQ(ierr);
	#else
		ierr = VecSetType(x, VECMPI); CHKERRQ(ierr);
	#endif
	ierr = VecSetFromOptions(x); CHKERRQ(ierr);

	/* some values */
	ierr = VecSet(x,1.0); CHKERRQ(ierr);

	mytimer.stop();
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- vector prepared in: %f s\n",mytimer.get_value_last()); CHKERRQ(ierr);

	/* maybe print the content of the vector ? */
	if(PRINT_VECTOR_CONTENT){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- Vector content: -------------\n"); CHKERRQ(ierr);
		ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	}

	/* save vector */
	mytimer.start();

	/* test SUM */
	double mysum = -1.0;
	mytimer.start();
	for(int i=0;i<N_TRIALS;i++){
		ierr = VecSum(x,&mysum); CHKERRQ(ierr);
	}
	mytimer.stop();
	double theory_sum = n;
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n- SUM info: ----------------\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- theoretical sum       : %f\n",theory_sum); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- computed sum          : %f\n",mysum); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- difference            : %g\n",std::abs(mysum-theory_sum)); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total time            : %g s\n", mytimer.get_value_last()); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  time            : %g s\n", mytimer.get_value_last()/(double)N_TRIALS); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);

	/* now create some funny index set and make real dummy things */
	IS myis;
	Vec xsub;
	ierr = ISCreateStride(PETSC_COMM_WORLD,1,0,1, &myis); CHKERRQ(ierr);
	ierr = VecGetSubVector(x, myis, &xsub); CHKERRQ(ierr);
	ierr = VecSet(xsub,2.0); CHKERRQ(ierr); /* yes, no change at all */
	ierr = VecRestoreSubVector(x, myis, &xsub); CHKERRQ(ierr);
	ierr = ISDestroy(&myis); CHKERRQ(ierr);
	
	/* compute sum */
	mysum = -1.0;
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


