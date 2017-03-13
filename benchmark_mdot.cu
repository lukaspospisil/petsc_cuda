/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* for measuring computation time */
#include "include/timer.h"

/* if everything fails, than try the simplest Hello World */
#define SAY_HELLO 0

/* I used this for testing purposes - to print content of vectors - to see that everything is OK */
#define PRINT_VECTOR_CONTENT 0

int main( int argc, char *argv[] )
{
	/* problem dimensions */
	int n = 10; /* length of vectors */
	int ntrials = 10; /* number of trials (to provide average time) */
	int m = 3; /* number of dot-products (I am computing <v1,v1>, <v1,v2>, <v1,v3>, ... <v1,vm>) */
	
	PetscErrorCode ierr; /* PETSc error */
	
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

	/* print info about benchmark */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"This is MDOT benchmark.\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - n          : %d\t\t(length of vectors)\n",n); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - ntrials    : %d\t\t(number of trials)\n",ntrials); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD," - m          : %d\t\t(number of dot-products)\n",m); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	
/* SAY HELLO TO WORLD - to check if everything is working */
	
	/* give info about MPI */
	int size, rank; /* size and rank of communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Hello world! */
	if(SAY_HELLO){
		ierr = PetscPrintf(MPI_COMM_WORLD,"- number of processors: %d\n",size); CHKERRQ(ierr);
		ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD," - hello from rank: %d\n",rank); CHKERRQ(ierr);
		ierr = PetscSynchronizedFlush(MPI_COMM_WORLD,NULL); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRQ(ierr);
	}

	/* timers */
	Timer mytimer;
	mytimer.restart();

/* CREATE GLOBAL VECTOR */
	mytimer.start();
	
	/* create first vector v1 (all other will have the same layout) */
	Vec v1;
	ierr = VecCreate(PETSC_COMM_WORLD,&v1); CHKERRQ(ierr);
	ierr = VecSetSizes(v1,PETSC_DECIDE,n); CHKERRQ(ierr);
	#ifdef USE_CUDA
		/* if we are using CUDA, it is a good idea to compute on GPU */
		ierr = VecSetType(v1, VECMPICUDA); CHKERRQ(ierr);
	#endif
	ierr = VecSetFromOptions(v1); CHKERRQ(ierr);

	/* some values (in my case I will try v_i = i) */
	ierr = VecSet(v1,1.0); CHKERRQ(ierr);

	/* prepare other vectors, i.e. array of vectors (because of m=?) */
	PetscScalar Mdots_val[m]; /* arrays of results of dot products */
	Vec Mdots_vec[m]; /* array of vectors */

	Mdots_vec[0] = v1; /* set first vector */

	/* prepare other vectors */
	for(int i=1;i<m;i++){
		ierr = VecDuplicate(v1, &(Mdots_vec[i])); CHKERRQ(ierr); 
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

/* COMPUTE DOT PRODUCTS */

	/* compute one dot product after another ("sequentially") */
	double dot_result;
	mytimer.start();
	for(int itrial=0;itrial<ntrials;itrial++){
		for(int i=0;i<m;i++){
			ierr = VecDot( Mdots_vec[0], Mdots_vec[i], &(Mdots_val[i])); CHKERRQ(ierr);
		}
	}
	mytimer.stop();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total sequential time : %f s\n", mytimer.get_value_last()); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  sequential time : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- results control       : " ); CHKERRQ(ierr);
	for(int i=0;i<m;i++){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ", Mdots_val[i]/(double)n); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);



	/* compute multiple dot products ("one shot") */
	mytimer.start();
	for(int itrial=0;itrial<ntrials;itrial++){
		ierr = VecMDot( Mdots_vec[0], m, Mdots_vec, Mdots_val); CHKERRQ(ierr);
	}
	mytimer.stop();

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total sequential time : %f s\n", mytimer.get_value_last()); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  sequential time : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- results control       : " ); CHKERRQ(ierr);
	for(int i=0;i<m;i++){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ", Mdots_val[i]/(double)n); CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRQ(ierr);




	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


