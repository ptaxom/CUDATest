#include "OperationHandler.cuh"


#include <stdio.h>

__global__ void matrixMult(const float *a, const float *b, int n, float *c) {
	int bX = blockIdx.x;
	int bY = blockIdx.y;

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	float sum = 0.0f;

	int indexOfA = n * BLOCK_SIZE * bY + n * tY;
	int indexOfB = BLOCK_SIZE * bX + tX;

	int indexOfC = n * BLOCK_SIZE * bY + BLOCK_SIZE * bX;


	for (int k = 0; k < n; k++)
		sum += a[indexOfA + k] * b[indexOfB + k * n];

	c[indexOfC + n * tY + tX] = sum;

}


__global__ void sharedMatrixMult(const float *a, const float *b, int n, float *c) {

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = blockIdx.y;

	int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;

	float sum = 0.0f, sum1 = 0.0f;

	
	for (int iA = aBegin, iB = bBegin; iA <= aEnd; iA += aStep, iB += bStep)
	{
		__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

		as[ty][tx] = a[iA + n * ty + tx];
		bs[ty][tx] = b[iB + n * ty + tx];

		as[ty + BLOCK_SIZE / 2][tx] = a[iA + n * (ty + BLOCK_SIZE / 2) + tx];
		bs[ty + BLOCK_SIZE / 2][tx] = b[iB + n * (ty + BLOCK_SIZE / 2) + tx];

		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++) {
			sum += as[ty][k] * bs[k][tx];
			sum1 += as[ty + BLOCK_SIZE / 2][k] * bs[k][tx];
		}
		__syncthreads();
	}

	

	c[aBegin + bBegin + ty * n + tx] = sum;
	c[aBegin + bBegin + (ty + BLOCK_SIZE / 2) * n + tx] = sum1;
}



__global__ void SMEM3(const float *A, const float *B, int n, float *C) {

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;

	int aStep = BLOCK_SIZE;
	int bStep = BLOCK_SIZE * n;

	int bBegin = BLOCK_SIZE * bx;

	float sum = 0.0f;

	for (int iA = aBegin, iB = bBegin; iA != aEnd; iA += aStep, iB += bStep) {

		__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

		as[ty][tx] = A[iA + n * ty + tx];
		bs[ty][tx] = B[iA + n * ty + tx];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += as[ty][k] * bs[k][tx];
		
		__syncthreads();
	}

	int iC = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	C[iC + n * ty + tx] = sum;
}

void multiplyerGPU(const float *A, const float *B, int N, float *C, int type) {
	int numBytes = N * N * sizeof(float);

	float *deviceA = nullptr;
	float *deviceB = nullptr;
	float *deviceC = nullptr;

	cudaMalloc((void**)&deviceA, numBytes);
	cudaMalloc((void**)&deviceB, numBytes);
	cudaMalloc((void**)&deviceC, numBytes);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMemcpy(deviceA, A, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, B, numBytes, cudaMemcpyHostToDevice);
	
	if (type == 0)
		matrixMult << <blocks, threads >> > (deviceA, deviceB, N, deviceC);
	if (type == 1) {
		SMEM3 << <blocks, threads >> > (deviceA, deviceB, N, deviceC);
	}
	if (type == 2) {
		dim3 threads4(BLOCK_SIZE, BLOCK_SIZE / 2);
		sharedMatrixMult << <blocks, threads4 >> > (deviceA, deviceB, N, deviceC);
	}


	cudaMemcpy(C, deviceC, numBytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);

	printf("type = %d; time spent executing by the GPU: %.2f millseconds\n",type, gpuTime);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);


}


void mem_test() {
	int numBytes = 128 * sizeof(double); // 1Kb 
	while (1) {
		double *arr;
		cudaMalloc((void**)&arr, numBytes);
	}
}