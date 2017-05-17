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
#define M 5
#define N_TRIALS 1000
#define PRINT_VECTOR_CONTENT 0

void compute_dots(int n, int ntrials, int m, Vec *Mdots_vec, double *Mdots_val) {
	PetscErrorCode ierr;

	Timer mytimer;
	mytimer.restart();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"### %d dot products\n", m); CHKERRV(ierr);

	/* compute dot product one after another ("sequentially") */
	mytimer.start();
	for(int itrial=0;itrial<ntrials;itrial++){
		for(int i=0;i<m;i++){
			ierr = VecDot( Mdots_vec[0], Mdots_vec[i], &(Mdots_val[i])); CHKERRV(ierr);
		}
	}
	mytimer.stop();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total time            : %f s\n", mytimer.get_value_last()); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  time            : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- results control       : " ); CHKERRV(ierr);
	for(int i=0;i<m;i++){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ", Mdots_val[i]/(double)n); CHKERRV(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRV(ierr);
}

void compute_mdot(int n, int ntrials, int m, Vec *Mdots_vec, double *Mdots_val) {
	PetscErrorCode ierr;

	Timer mytimer;
	mytimer.restart();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"### multiple %d dot-product\n", m); CHKERRV(ierr);

	/* compute multiple dot products ("one shot") */
	mytimer.start();
	for(int itrial=0;itrial<ntrials;itrial++){
		ierr = VecMDot( Mdots_vec[0], m, Mdots_vec, Mdots_val); CHKERRV(ierr);
	}
	mytimer.stop();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total time            : %f s\n", mytimer.get_value_last()); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  time            : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- results control       : " ); CHKERRV(ierr);
	for(int i=0;i<m;i++){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ", Mdots_val[i]/(double)n); CHKERRV(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRV(ierr);
}



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
	int m = M; /* number of dot-products (I am computing <v1,v1>, <v1,v2>, <v1,v3>, ... <v1,vm>) */
	
	/* print info about benchmark */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"This is MDOT test.\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - n          : %d\t\t(length of vectors)\n",n); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - ntrials    : %d\t\t(number of trials)\n",ntrials); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - m          : %d\t\t(number of dot-products)\n",m); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	
	Timer mytimer;
	mytimer.restart();
	mytimer.start();

	/* create first vector x1 (all other will have the same layout) */
	Vec x1;
	ierr = VecCreate(PETSC_COMM_WORLD,&x1); CHKERRQ(ierr);
	ierr = VecSetSizes(x1,PETSC_DECIDE,n); CHKERRQ(ierr);
	#ifdef USE_CUDA
		/* if we are using CUDA, it is a good idea to compute on GPU */
		ierr = VecSetType(x1, VECMPICUDA); CHKERRQ(ierr);
	#else
		ierr = VecSetType(x1, VECMPI); CHKERRQ(ierr);
	#endif
	ierr = VecSetFromOptions(x1); CHKERRQ(ierr);

	/* some values (in my case I will try v_i = i) */
	ierr = VecSet(x1,1.0); CHKERRQ(ierr);

	/* prepare other vectors, i.e. array of vectors (because of m=?) */
	PetscScalar Mdots_val[m]; /* arrays of results of dot products */
	Vec Mdots_vec[m]; /* array of vectors */

	Mdots_vec[0] = x1; /* set first vector */

	/* prepare other vectors */
	for(int i=1;i<m;i++){
		ierr = VecDuplicate(x1, &(Mdots_vec[i])); CHKERRQ(ierr); 
		ierr = VecSet(Mdots_vec[i],(PetscScalar)(i+1)); CHKERRQ(ierr);
	}

	mytimer.stop();
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- problem prepared in: %f s\n",mytimer.get_value_last()); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);

	/* maybe print the content of the vectors ? */
	if(PRINT_VECTOR_CONTENT){
		for(int i=0;i<m;i++){
			ierr = VecView(Mdots_vec[i], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
		}
	}

/* COMPUTE SEQUENTIALLY DOT PRODUCTS */
	compute_dots(n, ntrials, m, Mdots_vec, Mdots_val);

/* COMPUTE MULTIPLE DOT PRODUCTS */
	compute_mdot(n, ntrials, m, Mdots_vec, Mdots_val);

	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


