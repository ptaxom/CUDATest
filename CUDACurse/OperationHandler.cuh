#pragma once

#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16

//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif

#ifndef __CUDACC__
void __syncthreads();
#endif


void multiplyerGPU(const float *A, const float *B, int N, float *C, int type);


void mem_test();