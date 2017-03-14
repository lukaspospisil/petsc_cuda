/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* for measuring computation time */
#include "include/timer.h"

/* if everything fails, than try the simplest Hello World */
#define SAY_HELLO 0

/* I used this for testing purposes - to print content of vectors - to see that everything is OK */
#define PRINT_VECTOR_CONTENT 0


void say_hello() {
	PetscErrorCode ierr;

	/* give info about MPI */
	int size, rank; /* size and rank of communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Hello world! */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- number of processors: %d\n",size); CHKERRV(ierr);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD," - hello from rank: %d\n",rank); CHKERRV(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n"); CHKERRV(ierr);
		
}

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

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total sequential time : %f s\n", mytimer.get_value_last()); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  sequential time : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRV(ierr);
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

	ierr = PetscPrintf(PETSC_COMM_WORLD,"- total sequential time : %f s\n", mytimer.get_value_last()); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- avg.  sequential time : %f s\n", mytimer.get_value_last()/(double)ntrials); CHKERRV(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- results control       : " ); CHKERRV(ierr);
	for(int i=0;i<m;i++){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ", Mdots_val[i]/(double)n); CHKERRV(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n"); CHKERRV(ierr);
}



int main( int argc, char *argv[] )
{
	/* problem dimensions */
	int n = 1e7; /* length of vectors */
	int ntrials = 1e4; /* number of trials (to provide average time) */
	int m = 5; /* number of dot-products (I am computing <v1,v1>, <v1,v2>, <v1,v3>, ... <v1,vm>) */
	
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
	if(SAY_HELLO){
		say_hello();
	}

	Timer mytimer;
	mytimer.restart();
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

/* COMPUTE SEQUENTIALLY DOT PRODUCTS */
	compute_dots(n, ntrials, m, Mdots_vec, Mdots_val);

/* COMPUTE MULTIPLE DOT PRODUCTS */
	compute_mdot(n, ntrials, m, Mdots_vec, Mdots_val);

/* ensure that everything is on GPU */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"### TRANSFER PROBLEM TO GPU: "); CHKERRQ(ierr);
#ifdef USE_CUDA
	ierr = PetscPrintf(PETSC_COMM_WORLD,"YES (because USE_CUDA=ON)\n"); CHKERRQ(ierr);
	mytimer.start();
	for(int i=0;i<m;i++){
		ierr = VecCUDACopyToGPU(Mdots_vec[i]); CHKERRQ(ierr);
	}
	mytimer.stop();
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- problem transfered in: %f s\n",mytimer.get_value_last()); CHKERRQ(ierr);
#else
	ierr = PetscPrintf(PETSC_COMM_WORLD,"NO (because USE_CUDA=OFF)\n"); CHKERRQ(ierr);
#endif
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n"); CHKERRQ(ierr);


/* COMPUTE SEQUENTIALLY DOT PRODUCTS */
	compute_dots(n, ntrials, m, Mdots_vec, Mdots_val);

/* COMPUTE MULTIPLE DOT PRODUCTS */
	compute_mdot(n, ntrials, m, Mdots_vec, Mdots_val);

	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


