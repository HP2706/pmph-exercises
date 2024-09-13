#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS


__global__ void
replicate0(int tot_size, char* flags_d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < tot_size) {
        flags_d[gid] = 0;
    }
}



/* 
-- mkFlagArray: inspired by lecture notes page 48
let mkFlagArray 't [m]  
  (aoa_shp: [m]i64) 
  (zero: t) 
  (aoa_val: [m]t) : []t = 

  let shp_rot = map (\i -> 
    if i == 0 then 0
    else aoa_shp[i-1]
  ) (iota m)
  
  let shp_scn = scan (+) 0 shp_rot
  let aoa_len = if m == 0 then 0 
                else shp_scn[m-1] + aoa_shp[m-1]

  let shp_ind = map2 (\shp ind -> 
    if shp == 0 then -1
    else ind
  ) aoa_shp shp_scn

  in scatter (replicate aoa_len zero) shp_ind aoa_val

*/

__global__ void
mkFlags(
    int mat_rows, // number of rows 
    int* mat_shp_sc_d, // the scanned shape array
    char* flags_d // flags array
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < mat_rows) {
        int start_idx = 0;
        if (gid != 0) {
            start_idx = mat_shp_sc_d[gid - 1]; // we start at the end of the previous segment
        }
        int end_idx = mat_shp_sc_d[gid] - 1; // we end at the end of the current segment

        flags_d[start_idx] = 1;
        flags_d[end_idx] = 1;
    }
}


__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < tot_size) {
        tmp_pairs[gid] = mat_vals[gid] * vct[mat_inds[gid]];
    }
}


/* 
let last_indices = scan (+) 0 mat_shp
  
-- Extract the last element of each segmented sum
let row_sums = map (
\i -> if i == 0 then 
    scan_res[i]
else 
    scan_res[i - 1]
) last_indices
*/

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

