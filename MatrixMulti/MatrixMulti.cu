#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define TILE_SIZE 32

__global__ void matrixMulTiled(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float temp = 0.0f;

    // Loop over tiles needed to compute C[row,col]
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Wait for all threads to load tiles

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();  // Wait for computation to finish before loading new tile
    }

    // Write result
    if (row < n && col < n) {
        C[row * n + col] = temp;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;  
        h_B[i] = (float) (i / N);
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0] = %f\n", h_C[0]);
    printf("C[N*N-1] = %f\n", h_C[N*N-1]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}