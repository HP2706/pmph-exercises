#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#define GPU_RUNS 1 //100

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
        float temp = x / (x - 2.3);

        // we do this to avoid pow
        Y[i] = temp * temp * temp;
    }
}


int main(int argc, char** argv) {
    unsigned int N = 75341;

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    //randomInit(h_in, N);
    float* h_out = (float*) malloc(mem_size);
    float* h_out_seq = (float*) malloc(mem_size);

    // initialize the memory to all ones
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }

    naive_map(h_in, N, h_out_seq, N);


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
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        cuda_map<<<1, N>>>(d_in, d_out, N);
    }
  

    cudaDeviceSynchronize();
    for (int i = 0; i < GPU_RUNS; i++) {
        // execute the kernel a number of times;
        // to measure performance use a large N, e.g., 200000000,
        // and increase GPU_RUNS to 100 or more. 
    
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        
        for(int r = 0; r < GPU_RUNS; r++) {
            cuda_map<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
        }
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
        
    }
    cudaDeviceSynchronize();
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);


    validate<float>(h_out, h_out_seq, N, 0.0001);
    //printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}


/* int n_times = 100;

    float total_time = 0;

    // Seed the random number generator
    srand(time(NULL));

    for (int i = 0; i < n_times; i++) {
        float arr[75341];
        float result[75341];
        for (int j = 0; j < 75341; j++) {
            // Generate random float between 0 and 100
            arr[j] = (float)rand() / RAND_MAX * 100.0f;
        }

        clock_t start = clock();
        naive_map(arr, 75341, result, 75341);
        clock_t end = clock();

        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used / n_times;

        // Comment out or remove the following loop to avoid printing all results
        // for (int i = 0; i < 75341; i++) {
        //     printf("%f\n", result[i]);
        // }
    }
    printf("Average execution time: %f seconds\n", total_time);
    return 0; */