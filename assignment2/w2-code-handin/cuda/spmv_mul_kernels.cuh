#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS
#include <typeinfo>
#include <cstdio>


// prints the array on cpu for debugging before copying back to host
template <typename T>
void print_array(
    T* arr, 
    int size, 
    const char* name
) {
    T* h_arr = (T*)malloc(size * sizeof(T));
    CUDASSERT(cudaMemcpy(h_arr, arr, size * sizeof(T), cudaMemcpyDeviceToHost));
    printf("%s: ", name);
    for (int i = 0; i < size; i++) {
        if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
            printf("%f ", h_arr[i]);
        } else if (std::is_same<T, int>::value) {
            printf("%d ", h_arr[i]);
        } else if (std::is_same<T, char>::value) {
            printf("%d ", (int)h_arr[i]);
        } else if (std::is_same<T, bool>::value) {
            printf("%d ", (int)h_arr[i]);
        } else if (std::is_same<T, char>::value) {
            printf("%d ", (char)h_arr[i]);
        } else {
            printf("%d ", (int)h_arr[i]);
        }


    }
    CUDASSERT(cudaMemcpy(arr, h_arr, size * sizeof(T), cudaMemcpyHostToDevice));
    printf("\n");
    free(h_arr);
}



__global__ void replicate0(int tot_size, char* flags_d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < tot_size) {
        flags_d[gid] = 0;
    }
}

__global__ void
mkFlags(
    int mat_rows, // number of rows 
    int* mat_shp_sc_d, // the scanned shape array
    char* flags_d // flags array
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < mat_rows) {
        int start_idx = mat_shp_sc_d[gid]; // we start at the end of the previous segment
        flags_d[start_idx] = 1;
    }
}


__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < tot_size) {
        tmp_pairs[gid] = mat_vals[gid] * vct[mat_inds[gid]];
    }
}

__global__ void
select_last_in_sgm(
    int mat_rows, 
    int* mat_shp_sc_d, // the shape array segmented scan
    float* tmp_scan, // the scan
    float* res_vct_d // store the result
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < mat_rows) {
        res_vct_d[gid] = tmp_scan[mat_shp_sc_d[gid] - 1]; // the read from temp_scan the last element of each segment
    }
}

#endif // SPMV_MUL_KERNELS

