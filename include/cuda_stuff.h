#ifndef CUDA_STUFF_H
#define	CUDA_STUFF_H

#include <cuda.h>
#include "petsccuda.h"
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

/* cuda error check */ 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"\n\x1B[31mCUDA error:\x1B[0m %s %s \x1B[33m%d\x1B[0m\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void warm_up_kernel(){

}

void warm_up_cuda(){
	PetscErrorCode ierr; 	
	warm_up_kernel<<<1, 1>>>();
	gpuErrchk( cudaDeviceSynchronize() );
	ierr = PetscBarrier(NULL); CHKERRV(ierr);
}


#endif
