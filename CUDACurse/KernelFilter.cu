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

	__syncthreads();

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

	int pixel = (((int)r) << 16) +
		(((int)g) << 8) +
		(int)b;

	__syncthreads();

	output[(tileX + tX + offset) * rows + (tileY + tY + offset)] = pixel;

}