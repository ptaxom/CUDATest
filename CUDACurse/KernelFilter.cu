#include "OperationHandler.cuh"


#include <stdio.h>


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


__global__ void kernelFilter(const int* image, const float *kernel,
							 const int kernelHalf, const int offset,
							 const int rows, const int cols,
								   int* output)
{


	int tileX = (BLOCK_SIZE_X - kernelHalf * 2) * blockIdx.x;
	int tileY = (BLOCK_SIZE_Y - kernelHalf * 2) * blockIdx.y;

	int tX = threadIdx.x;
	int tY = threadIdx.y;
	
	__shared__ int imgTile[BLOCK_SIZE_X][BLOCK_SIZE_Y];

	int idX = tileX + tX;
	int idY = tileY + tY;

	imgTile[tX][tY] = image[idX * rows + idY];

	//__syncthreads();

	if (tX < kernelHalf || tY < kernelHalf || tX >= BLOCK_SIZE_X - kernelHalf || tX >= BLOCK_SIZE_X - kernelHalf)
		return;

	int kernelSize = 2 * kernelHalf + 1;

	float r = 0, g = 0, b = 0;
	for (int i = -kernelHalf; i < kernelHalf; i++)
		for (int j = -kernelHalf; j < kernelHalf; j++)
		{
			int pixel = imgTile[tX + i][tY + j];
			float pR = (float)((pixel >> 16) && 0x000000ff);
			float pG = (float)((pixel >> 8) && 0x000000ff);
			float pB = (float)(pixel && 0x000000ff);
			float kern = kernel[(i + kernelHalf) * kernelSize + j + kernelHalf];
			
			r += kern * pR;
			g += kern * pG;
			b += kern * pB;
		}

	if (r < 0) r = 0; if (r > 255) r = 255;
	if (g < 0) g = 0; if (g > 255) g = 255;
	if (b < 0) b = 0; if (b > 255) b = 255;

	int pixel = (((int)r) << 16) +
		(((int)g) << 8) +
		(int)b;

	//__syncthreads();

	output[(tileX + tX + offset) * rows + (tileY + tY + offset)] = pixel;

}

int min2(int a, int b)
{
	return a < b ? a : b;
}

void kernelGPU(const int* image, const float *kernel,
				   const int kernelHalf,
				   const int cols, const int rows,
				   int* output)
{
	int numBytes = cols * rows * sizeof(int);
	int kernelSizeBytes = (kernelHalf * 2 + 1) * (kernelHalf * 2 + 1);

	int *deviceImage = nullptr;
	int *outputImage = nullptr;
	float *deviceKernel = nullptr;

	cudaMalloc((void**)&deviceImage, numBytes);
	cudaMalloc((void**)&outputImage, numBytes);
	cudaMalloc((void**)&deviceKernel, kernelSizeBytes);

	dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks(rows / threads.x, cols / threads.y);

	int offset = min2((rows % threads.x) / 2, (cols % threads.y) / 2);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMemcpy(deviceImage, image, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceKernel, kernel, numBytes, cudaMemcpyHostToDevice);

	kernelFilter <<<blocks, threads >>> (image, kernel, kernelHalf,offset, rows, cols, output);


	cudaMemcpy(output, outputImage, numBytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);

	printf(" time spent executing by the GPU: %.2f millseconds\n", gpuTime);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(outputImage);
	cudaFree(deviceImage);
	cudaFree(deviceKernel);


}