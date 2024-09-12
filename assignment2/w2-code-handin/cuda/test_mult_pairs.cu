#include <iostream>
#include <vector>
#include <random>
#include "spmv_mul_kernels.cuh"
#include "host_skel.cuh"


// Helper function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int vct_size = 10;
    int mat_rows = 10;

    const int tot_size = vct_size * mat_rows;
    const int block_size = 1024;
    // Remove unused variable
    const int num_blocks_shp = 128;
    const int num_blocks = (tot_size + block_size - 1) / block_size;

    int   *mat_shp_d, *mat_shp_sc_d, *d_tmp_int;
    char  *flags_d;

    // Initialize host arrays
    int* mat_shp_h = new int[mat_rows];
    for (int i = 0; i < mat_rows; i++) {
        mat_shp_h[i] = i + 1; // Example initialization
    }

    cudaMalloc((void**)&mat_shp_d,    mat_rows*sizeof(int ));
    cudaMalloc((void**)&mat_shp_sc_d, mat_rows*sizeof(int ));
    cudaMalloc((void**)&flags_d,      tot_size*sizeof(char));
    cudaMalloc((void**)&d_tmp_int,   MAX_BLOCK*sizeof(int));
    cudaCheckError();

    // Copy initialized data to device
    cudaMemcpy(mat_shp_d, mat_shp_h, mat_rows*sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    char* flags_cpu = new char[tot_size];


    // 2. create an array of zeros
    replicate0<<< num_blocks, block_size >>> ( tot_size, flags_d );
    cudaCheckError();

    // 3. scatter the flag array
    // Uncomment and fix if needed

    int mat_shp_flags[10] = {0, 2, 0, 0, 5, 0, 0, 0, 0, 1};
    
    // Allocate and copy mat_shp_flags to device
    int *d_mat_shp_flags;
    cudaMalloc((void**)&d_mat_shp_flags, mat_rows * sizeof(int));
    cudaMemcpy(d_mat_shp_flags, mat_shp_flags, mat_rows * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate cumulative sum (scan) of mat_shp_flags
    thrust::device_ptr<int> dev_ptr(d_mat_shp_flags);
    thrust::inclusive_scan(dev_ptr, dev_ptr + mat_rows, dev_ptr);

    // Launch kernel with correct arguments
    mkFlags<<< num_blocks_shp, block_size >>> (mat_rows, d_mat_shp_flags, flags_d);
    cudaCheckError();

    cudaMemcpy(flags_cpu, flags_d, tot_size*sizeof(char), cudaMemcpyDeviceToHost);
    cudaCheckError();

    for (int i = 0; i < tot_size; i++) {
        std::cout << (int)flags_cpu[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] mat_shp_h;
    delete[] flags_cpu;
    cudaFree(mat_shp_d);
    cudaFree(mat_shp_sc_d);
    cudaFree(flags_d);
    cudaFree(d_tmp_int);
    cudaFree(d_mat_shp_flags);

    return 0;
}