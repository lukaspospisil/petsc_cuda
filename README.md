Simple benchmarks for PETSc framework 
(especially for comparing operations performed by CPU/GPU on Piz Daint supercomputer)

How to compile:
- load right modules (i.e. set PETSC_DIR, PETSC_ARCH, ...), on Piz Daint using 
```
source util/module_load_daint_*
```
- prepare build directory and call cmake 
```
mkdir build
cd dir
cmake ..
```
- call `make` to compile what you want to compile
- there is `batch` folder with sample batch files for SLURM on Piz Daint, use & enjoy them! Just be sure that you edit them and set right paths to set-set-modules-files



```
# cmake options:
# -DBENCHMARK_***=ON 		-> compile benchmark *** 
# -DFIND_PETSC=ON			-> use functions from Jed Brown to set Petsc variables (turn off if using Petsc from modules)
# -DUSE_CUDA=ON				-> compile with GPU
#
```
