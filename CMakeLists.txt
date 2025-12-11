cmake_minimum_required(VERSION 3.12)
project(matmul C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include_directories("include")
file(GLOB SRC "src/*.c")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/experiment.c")

option(USE_MPI "Build with MPI support" OFF)

# CBLAS discovery
if(CBLAS_LIBRARIES AND EXISTS ${CBLAS_LIBRARIES})
    set(_CBLAS_LIB_HINT ${CBLAS_LIBRARIES})
endif()

find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${CBLAS_INCLUDE_DIR})
find_library(CBLAS_LIBRARIES NAMES cblas openblas HINTS ${_CBLAS_LIB_HINT})

if(NOT CBLAS_INCLUDE_DIR OR NOT CBLAS_LIBRARIES)
    message(FATAL_ERROR "CBLAS not found. Provide -DCBLAS_INCLUDE_DIR and -DCBLAS_LIBRARIES.")
endif()

include_directories(${CBLAS_INCLUDE_DIR})

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(OPENMPI)
    add_executable(matmul ${SRC})
    # link MPI, CBLAS/OpenBLAS and the math library (libm)
    target_link_libraries(matmul PRIVATE MPI::MPI_C ${CBLAS_LIBRARIES} m)

else()
    add_executable(matmul ${SRC})
    # link CBLAS/OpenBLAS and the math library (libm)
    target_link_libraries(matmul PRIVATE ${CBLAS_LIBRARIES} m)

endif()

# Build types: Release (default), Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS_DEBUG   "-g -Wall -Wextra -Werror")