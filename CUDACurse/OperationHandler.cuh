#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16




void multiplyerGPU(const float *A, const float *B, int N, float *C);
