project(PETSC_CUDA_TEST)
cmake_minimum_required(VERSION 2.8)

#decide which example to compile
# to turn on/off these variables call "cmake -DBENCHMARK_CPUSEQ=ON .." or "cmake -DBENCHMARK_CPUSEQ=OFF .."
option(BENCHMARK_CPUSEQ "BENCHMARK_CPUSEQ" ON)
option(BENCHMARK_CUDASEQ "BENCHMARK_CUDASEQ" OFF)


# CMAKE: include cmake functions
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/util/cmake/" ${CMAKE_MODULE_PATH})

# PETSc: defined paths in ENV? on PIZ Daint not important - after loading modules everything is prepared 
option(FIND_PETSC "FIND_PETSC" OFF)
if(${FIND_PETSC})
	# magic function from Jed Brown
	find_package(PETSc)
#	set(CMAKE_CXX_COMPILER "mpicxx")
endif()

# give some info about Petsc
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "PETSC_DIR = $ENV{PETSC_DIR}")
message(STATUS "PETSC_ARCH = $ENV{PETSC_DIR}")
message(STATUS "PETSC_INCLUDES = ${PETSC_INCLUDES}")

# PETSc: include petsc stuff
include_directories(${PETSC_INCLUDES})

# maybe also some CUDA stuff ?? (following code originaly found in MinLin)
#	include(FindCUDA)
#	set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
#	set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
#	string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
#	set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
#	string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options


if(${BENCHMARK_CPUSEQ})
	ADD_EXECUTABLE(benchmark_cpuseq benchmark_cpuseq.cpp)
	TARGET_LINK_LIBRARIES(benchmark_cpuseq ${PETSC_LIBRARIES})
endif()

if(${BENCHMARK_CUDASEQ})
# todo: here will be compilation of petsc-cuda sample
# compile with nvcc? 

#	cuda_add_executable(benchmark_cudaseq benchmark_cudaseq.cu
#			OPTIONS "-arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
#			DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
#	target_link_libraries(benchmark_cudaseq ${LIBRARIES_DEF})

endif()
