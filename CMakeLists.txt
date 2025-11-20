cmake_minimum_required(VERSION 3.12)
project(matmul C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include_directories("include")
file(GLOB SRC "src/*.c")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/experiment.c")

option(USE_MPI "Build with MPI support" OFF)

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(OPENMPI)
    add_executable(matmul ${SRC})
    # link MPI, OpenBLAS and the math library (libm)
    target_link_libraries(matmul PRIVATE MPI::MPI_C openblas m)

else()
    add_executable(matmul ${SRC})
    # link OpenBLAS and the math library (libm)
    target_link_libraries(matmul PRIVATE openblas m)

endif()

# Build types: Release (default), Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS_DEBUG   "-g -Wall -Wextra -Werror")