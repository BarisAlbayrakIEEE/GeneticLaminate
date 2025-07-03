#!/usr/bin/env bash
# Bash script to compile genetic_stacking.cu with nvcc

set -euo pipefail

CUDA_PATH=/usr/local/cuda
export PATH=${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}

SM_ARCH=sm_50
STD_VER=c++11 # not used, defaulted by nvcc
SRC=genetic_stacking.cu
OUT=genetic_stacking.out

nvcc -ccbin gcc \
  -arch=${SM_ARCH} \
  -O3 \
  -rdc=true \
  -Wno-deprecated-declarations \
  ${SRC} -o ${OUT}

echo "Built ${OUT} for compute capability ${SM_ARCH}"
