# options:
# -DBENCHMARK_***=ON 		-> compile benchmark *** 
# -DFIND_PETSC=ON			-> use functions from Jed Brown to set Petsc variables (turn off if using Petsc from modules)
# -DUSE_CUDA=ON				-> compile with GPU
#

project(PETSC_CUDA_TEST)
cmake_minimum_required(VERSION 2.8)

option(BENCHMARK_SEQ "BENCHMARK_SEQ" ON)
option(FIND_PETSC "FIND_PETSC" ON)
option(USE_CUDA "USE_CUDA" OFF)

# CMAKE: include cmake functions
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/util/cmake/" ${CMAKE_MODULE_PATH})

# PETSc: defined paths in ENV? on PIZ Daint not important - after loading modules everything is prepared 
if(${FIND_PETSC})
	# magic function from Jed Brown
	find_package(PETSc)
#	set(CMAKE_CXX_COMPILER "mpicxx")
endif()

# give some info about Petsc
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "PETSC_DIR = $ENV{PETSC_DIR}")
message(STATUS "PETSC_ARCH = $ENV{PETSC_ARCH}")
message(STATUS "PETSC_INCLUDES = ${PETSC_INCLUDES}")
message(STATUS "PETSC_LIBRARIES = ${PETSC_LIBRARIES}")
message(STATUS "FIND_PETSC = ${FIND_PETSC}")
message(STATUS "USE_CUDA = ${USE_CUDA}")


# PETSc: include petsc stuff
include_directories(${PETSC_INCLUDES})

# maybe also some CUDA stuff ?? (following code originaly found in MinLin)
if(${USE_CUDA})
	include(FindCUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
	set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
	string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
	set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
	string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options

	# send USE_CUDA to compiled code
	set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} -DUSE_CUDA")
endif()

if(${BENCHMARK_SEQ})
	if(${USE_CUDA})
		# compile using cuda compiler
		cuda_add_executable(benchmark_seq benchmark_seq.cu
			OPTIONS "-arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
			DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
		target_link_libraries(benchmark_seq ${PETSC_LIBRARIES})
	else ()
		# compile using standart c++ compiler
		add_executable(benchmark_seq benchmark_seq.cpp)
		target_link_libraries(benchmark_seq ${PETSC_LIBRARIES})
	endif ()
endif()
