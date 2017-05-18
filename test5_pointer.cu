/* include c++ headers */
#include <iostream>

/* include petsc */
#include "petsc.h"
#include "mpi.h"

/* for measuring computation time */
#include "include/timer.h"
#ifdef USE_CUDA
	/* some cuda helpers */
	#include "include/cuda_stuff.h"
#endif

/* X_SIZE=n*MPI_COMM_SIZE */
#define X_SIZE 1e6
#define N_TRIALS 1000
#define PRINT_VECTOR_CONTENT 0


template<class VectorBase>
class GeneralVector : public VectorBase {
	public:
		/* call original constructor with one argument */
		template<class ArgType> GeneralVector(ArgType arg) : VectorBase(arg) {
		}	
	
		std::string who_am_I(){
			return "I am GeneralVector";
		}
};

class PetscVector {
	private:
		Vec inner_vector;
	public:
		PetscVector(const Vec &new_inner_vector){
			this->inner_vector = new_inner_vector;
		}
	
		Vec get_vector() const{
			return inner_vector;
		}

		std::string who_am_I(){
			return "I am PetscVector";
		}
	
};


double test_sum(Vec &x){
	PetscErrorCode ierr; 

	Timer mytimer;
	mytimer.restart();

	double mysum = -1.0;
	mytimer.start();
	for(int i=0;i<N_TRIALS;i++){
		ierr = VecSum(x,&mysum); CHKERRQ(ierr);
	}
	mytimer.stop();

	return mytimer.get_value_last();
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
	
	/* print info about benchmark */
	ierr = PetscPrintf(PETSC_COMM_WORLD,"This is POINTER test.\n"); CHKERRQ(ierr);
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

/* perform some dummy things */
	GeneralVector<PetscVector> *myvector = new GeneralVector<PetscVector>(x);

	/* present yourself */
	std::cout << std::endl;
	std::cout << " - Who am I?: " << myvector->who_am_I() << std::endl;

	/* now the idea - get Vec from myvector */
	Vec y = myvector->get_vector(); /* can be called because of inheritance */


	/* try to sum to see immediatelly if we are computing on CPU or GPU */
	double mytime;

	mytime = test_sum(x);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 1. test: %f\n",mytime); CHKERRQ(ierr);

	mytime = test_sum(y);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 2. test: %f\n",mytime); CHKERRQ(ierr);
	

	
/*	
	 mything(x);
	VecEnvelope *pointer_to_mything = &mything;
	VecEnvelope *pointer_to_mything2 = new VecEnvelope(x);
	double mytime;
	
	mytime = test_sum(x);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 1. test: %f\n",mytime); CHKERRQ(ierr);

	Vec x2 = mything.get_vector();
	mytime = test_sum(x2);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 2. test: %f\n",mytime); CHKERRQ(ierr);

	Vec x3 = pointer_to_mything->get_vector();
	mytime = test_sum(x3);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 3. test: %f\n",mytime); CHKERRQ(ierr);

	Vec x4 = pointer_to_mything2->get_vector();
	mytime = test_sum(x4);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"- 4. test: %f\n",mytime); CHKERRQ(ierr);
*/

	/* destroy vector */
	ierr = VecDestroy(&x); CHKERRQ(ierr);

	/* finalize Petsc */
	PetscFinalize();

	return 0;
}


