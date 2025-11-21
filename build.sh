#!/bin/bash
set -e

module purge
module load gcc/9.4.0
module load openblas/0.3.21-gcc-9.4.0
module load openmpi/4.1.1-gcc-9.4.0

module load cmake

BUILD=build
SRC=$(pwd)

rm -rf "$BUILD"

cmake -B $BUILD -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=1
cmake --build $BUILD
