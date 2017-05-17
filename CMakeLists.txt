# options:
# -DTEST*_***=ON 			-> compile test *** 
# -DFIND_PETSC=ON			-> use functions from Jed Brown to set Petsc variables (turn off if using Petsc from modules)
# -DUSE_CUDA=ON				-> compile with GPU
#
# on my computer (because I am compiler-lama), I am compiling with
# cmake -DFIND_PETSC=ON -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx ..
#

project(PETSC_CUDA_TEST)
cmake_minimum_required(VERSION 2.8)

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

# add some CUDA stuff (following code originaly found in MinLin)
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

# for simple print ON/OF
macro(PRINTINFO_ONOFF name value)
	message(" ${name} : ${value}")
endmacro()

macro(ADD_EXECUTABLE_CPP name filename)
	# compile using standart c++ compiler
	add_executable(${name} ${filename})
	target_link_libraries(${name} ${PETSC_LIBRARIES})
endmacro()

macro(ADD_EXECUTABLE_CUDA name filename)
	# compile using cuda compiler
	cuda_add_executable(${name} ${filename}
		OPTIONS "-arch=sm_60 --compiler-options \"${CUDA_CXX_FLAGS}\""
			DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
	target_link_libraries(${name} ${PETSC_LIBRARIES})
endmacro()


# ---------------------- MY BENCHMARKS -----------------------
message("\nThe list of my tests:")

# copy sample batchscripts for computation oo Piz Daint (i.e. SLURM files)
make_directory("batch")
make_directory("batch_out")
file(COPY "batch/" DESTINATION "batch" FILES_MATCHING PATTERN "*")


option(TEST1_SUM "TEST1_SUM" OFF)
printinfo_onoff(" TEST1_SUM                   " "${TEST1_SUM}")
if(${TEST1_SUM})
	if(${USE_CUDA})
		add_executable_cuda(test1_sum test1_sum.cu)
	else ()
		add_executable_cpp(test1_sum test1_sum.cpp)
	endif ()
endif()

option(TEST2_MDOT "TEST2_MDOT" OFF)
printinfo_onoff(" TEST2_MDOT                  " "${TEST2_MDOT}")
if(${TEST2_MDOT})
	if(${USE_CUDA})
		add_executable_cuda(test2_mdot test2_mdot.cu)
	else ()
		add_executable_cpp(test2_mdot test2_mdot.cpp)
	endif ()
endif()

option(TEST3_LOAD "TEST3_LOAD" OFF)
printinfo_onoff(" TEST3_LOAD                  " "${TEST3_LOAD}")
if(${TEST3_LOAD})
	if(${USE_CUDA})
		add_executable_cuda(test3_load test3_load.cu)
	else ()
		add_executable_cpp(test3_load test3_load.cpp)
	endif ()
endif()

option(TEST4_IS "TEST4_IS" OFF)
printinfo_onoff(" TEST4_IS                    " "${TEST4_IS}")
if(${TEST4_IS})
	if(${USE_CUDA})
		add_executable_cuda(test4_is test4_is.cu)
	else ()
		add_executable_cpp(test4_is test4_is.cpp)
	endif ()
endif()

