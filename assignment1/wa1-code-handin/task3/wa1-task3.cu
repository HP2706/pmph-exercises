#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#define GPU_RUNS 100 
#define N 75341
#define CPU_RUNS // only one run for cpu to avoid caching effects on measurements

//$f(x) = (\frac{x}{x-2.3})^3$ to an array of size 75341
void naive_map(float *arr, int n, float *result, int result_size) {
    if (result_size < n) {
        printf("Result size is less than n\n");
        assert(0);
    }
    for (int i = 0; i < n; i++) {
        float x = arr[i];
        float temp = x / (x - 2.3);
        result[i] = temp * temp * temp;
    }
}


__global__ void cuda_map(float* X, float* Y, int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = X[i]; // Load the input element
        float temp = __fdividef(x, x - 2.3f);

        // we do this to avoid pow
        Y[i] = temp * temp * temp;
    }
}

#define BLOCK_SIZE 32

__global__ void cuda_map_improved(float* X, float* Y, int n) {
    __shared__ float s_data[BLOCK_SIZE];  // Adjust size based on your block size
    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory
    if (i < n) {
        s_data[tid] = X[i];
    }
    __syncthreads();

    if (i < n) {
        float x = s_data[tid];
        float temp = __fdividef(x, x - 2.3f);  // Fast float division

        // Unrolled multiplication
        float temp2 = temp * temp;
        Y[i] = temp2 * temp;
    }
}


int main(int argc, char** argv) {
    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    //randomInit(h_in, N);
    float* h_out = (float*) malloc(mem_size);
    float* h_out_seq = (float*) malloc(mem_size);

    // initialize the memory to all ones
    randomInit(h_in, N);

    // time the naive map
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    naive_map(h_in, N, h_out_seq, N);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    double elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
    printf("Naive map took %.2f microseconds\n", elapsed);
    double gigabytespersec_naive = (2.0 * N * 4.0) / (elapsed * 1000.0);
    printf("Naive map took %.2f GB/s\n", gigabytespersec_naive);


    // use the first CUDA device:
    cudaSetDevice(0);

    // allocate device memory
    float* d_in;
    float* d_out;




    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // Calculate the grid and block dimensions
    int BlockSize = 32;
    int blocksPerGrid = (N + BlockSize - 1) / BlockSize;



    // a small number of dry runs
    for(int r = 0; r < 100; r++) {
        cuda_map_improved<<<blocksPerGrid, BlockSize>>>(d_in, d_out, N);
    }
  

    double total_elapsed = 0;
    for (int i = 0; i < GPU_RUNS; i++) {
        // execute the kernel a number of times;
        // to measure performance use a large N, e.g., 200000000,
        // and increase GPU_RUNS to 100 or more. 
    
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
        
        cuda_map_improved<<<blocksPerGrid, BlockSize>>>(d_in, d_out, N);
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
        total_elapsed += elapsed;
    }

    double avg_elapsed = total_elapsed / GPU_RUNS;
    double gigabytespersec = (2.0 * N * 4.0) / (avg_elapsed * 1000.0);


    printf(
        "The kernel took on average %.2f microseconds. GB/sec: %.2f over %d runs\n", 
        avg_elapsed, gigabytespersec, GPU_RUNS
    );
    cudaDeviceSynchronize();
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    
    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);


    validate<float>(h_out, h_out_seq, N, 0.0001);
    //validateExact<float>(h_out, h_out_seq, N);
    //printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}
