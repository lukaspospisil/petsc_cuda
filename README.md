Simple benchmarks for PETSc framework 
(especially for comparing operations performed by CPU/GPU on Piz Daint supercomputer)


- the code is compiled using "cmake" (with magic "FindPetsc" function from Jed Brown), example for Piz Daint follows. Here the code has to (should be) compiled and run on special "$SCRATCH" folder.

```
ssh pospisil@ela.cscs.ch
ssh daint

mkdir soft
cd soft
git clone https://github.com/lukaspospisil/petsc_cuda.git
```

- for compulation on CPU use:
```
cd $SCRATCH
mkdir build_cpu
cd build_cpu
source ~/soft/util/module_load_daint
cmake -DUSE_CUDA=OFF -DFIND_PETSC=OFF -DTEST1_SUM=ON -DTEST2_MDOT=ON -DTEST3_LOAD=ON -DTEST4_IS=ON ~/soft/petsc_cuda/
make
sbatch --account=c11 --constraint=gpu batch/test1_cpu2.batch
sbatch --account=c11 --constraint=gpu batch/test2_cpu2.batch
sbatch --account=c11 --constraint=gpu batch/test3_cpu2.batch
```

- for compilation on GPU use:
```
cd $SCRATCH
mkdir build_gpu
cd build_gpu
source ~/soft/util/module_load_daint_sandbox
cmake -DUSE_CUDA=ON -DFIND_PETSC=ON -DTEST1_SUM=ON -DTEST2_MDOT=ON -DTEST3_LOAD=ON -DTEST4_IS=ON ~/soft/petsc_cuda/
make
sbatch --account=c11 --constraint=gpu batch/test1_gpu2.batch
sbatch --account=c11 --constraint=gpu batch/test1_gpu1_does_not_work.batch
sbatch --account=c11 --constraint=gpu batch/test2_gpu2.batch
sbatch --account=c11 --constraint=gpu batch/test3_gpu2.batch
```


TEST1:
- create VECMPI (VECMPICUDA if USE_GPU=ON)
- get local array and fill vector with x[i]=q^i, i=0,\dots,n-1 using FOR cycle (or call kernel if USE_CUDA=ON)
- compute several times SUM of vector, compare it with theoretical results (it is geometrical sequence), provide time measurement
- OBSERVED PETSC BUG: if we are running more MPI processes on one node (see batch/test1_gpu1_does_not_work.batch) then (probably) all of them are trying to allocate GPU for themself:
```
[1]PETSC ERROR: --------------------- Error Message --------------------------------------------------------------
[1]PETSC ERROR: Error in external library
[1]PETSC ERROR: CUBLAS error 1
[1]PETSC ERROR: See http://www.mcs.anl.gov/petsc/documentation/faq.html for trouble shooting.
[1]PETSC ERROR: Petsc Release Version 3.7.4, unknown 
[1]PETSC ERROR: /scratch/snx3000/pospisil/petsc_cuda/./test1_sum on a arch-gnu-xc30-daint-cuda named nid06050 by pospisil Wed May 17 12:53:40 2017
[1]PETSC ERROR: Configure options --with-cc=cc --with-cxx=CC --with-fc=ftn COPTFLAGS=-O3 CXXOPTFLAGS=-03FOPTFLAGS=-03--with-clib-autodetect=0 --with-cxxlib-autodetect=0 --with-fortranlib-autodetect=0 --with-shared-libraries=0 --with-debugging=0 --with-valgrind=0 --known-mpi-shared-libraries=1 --with-x=0 PETSC_ARCH=arch-gnu-xc30-daint-cuda --with-cuda=1 --with-cuda-arch=sm_60 --with-cuda-dir=/opt/nvidia/cudatoolkit8.0/8.0.44_GA_2.2.7_g4a6c213-2.1 --CUDAFLAGS=-I/opt/cray/pe/mpt/7.5.0/gni/mpich-gnu/5.1/include/
[1]PETSC ERROR: #1 PetscInitialize() line 927 in /apps/daint/UES/6.0.UP02/sandbox-ws/petsc-maint/src/sys/objects/pinit.c
```

TEST2:
- create array of VECMPI (VECMPICUDA if USE_GPU=ON)
```
x1 = ones(n,1);
x2 = 2*ones(n,1);
x3 = 3*ones(n,1);
x4 = 4*ones(n,1);
....
```
- compute several times MDOT x1*x?, ?=1,2,3,4,\dots (which is acctually the same as sums of these vectors)
- compare with theoretical results, provide time measurement
- OBSERVED PETSC BUG: computing time si really strange - in GPU case the MDOT is slower than sequential DOTs

TEST3:
- create VECMPI (VECMPICUDA) vector x[i]=1, save it to disk, destroy vector
- load vector from disk to VECMPI (VECMPICUDA if USE_GPU=ON)
- compute sum and measure time
- OBSERVED PETSC BUG: actually, this test works... I assumed that if I load the vector, then this vector will be on CPU.. but the bug is somewhere else

TEST4:
- create VECMPI (VECMPICUDA) vector x[i]=1,
- compute sum and measure time
- non-local getsubvector using index set
- OBSERVED PETSC BUG: actually, this test also works... :(



