#include <iostream>
#include "OperationHandler.cuh"

void printMatrix(float *matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			printf("%.2f ", matrix[i*size + j]);
		printf("\n");
	}
}

int main() {
	int N = 1024*4;
	float *A = new float[N*N];
	float *B = new float[N*N];
	float *C = new float[N*N];

	for (int i = 0; i < N*N; i++) {
		A[i] = (rand() % 1024) / 32;
		B[i] = 0;
	}

	for (int i = 0; i < N; i++)
		B[i*N + i] = 1;
	
	multiplyerGPU(A, B, N, C);

	delete A;
	delete B;
	delete C;

	return 0;
}