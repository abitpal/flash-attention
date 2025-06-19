// #include "utils.h"
#include "flash_attention_kernel.h"
#include "consts.h"
#include <assert.h>
#include <cstdio> 

// constexpr int get_sram_size() {
//     int device = 0; 
//     cudaDeviceProp prop = NULL; 

//     cudaGetDevice(&device);
//     cudaGetDeviceProperties(&prop, device);
//     return prop.sharedMemPerBlock; 
// }

const int d_k = 64;  // template this
const int sqrt_d_k = 16; 
// const int sram_size_limit = 49152 / sizeof(float); 
// const int max_b_r = (sram_size_limit / (d_k * 4));  // block rows (query block size)
// const int b_c = b_r; 
// const int block_x = min(32, b_r); 
// const int block_y = 1024 / block_x; 

__device__ float operator*(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; 
}


__global__ 
void flash_attn_forward(
    float* Q, float* K, float* V, float* O, const int b_c, const int b_r, const int t_c, 
    const int t_r, const int n_seq_k, const int n_seq_q,
    const int _d_k, const float scaling_factor) {
    int thread_x = threadIdx.x, thread_y = threadIdx.y, thread_z = threadIdx.z; 
    int tcount_x = blockDim.x, tcount_y = blockDim.y, tcount_z = blockDim.z; 
    int tid = thread_y * tcount_x + tcount_x; 
    int batch_idx = blockIdx.x, head_idx = blockIdx.y;  // batch and head index
    int n_head = gridDim.y; 

    extern __shared__ float sram []; 
    float* q_i = sram; 
    float* k_i = sram + b_r * d_k; 
    float* v_i = k_i + b_c * d_k; 
    float* o_i = v_i + b_c * d_k; 

    Q += (batch_idx * n_head * n_seq_q * d_k) + head_idx * (n_seq_q * d_k);
    K += (batch_idx * n_head * n_seq_k * d_k) + head_idx * (n_seq_k * d_k);
    V += (batch_idx * n_head * n_seq_k * d_k) + head_idx * (n_seq_k * d_k);

    // https://siboehm.com/articles/22/CUDA-MMM Global Memory Coalescing
    for (int q_idx = 0; q_idx < t_r; q_idx++) {
        int true_br = min(n_seq_q, (q_idx + 1) * b_r) - b_r * q_idx; 
        for (int i = thread_z * tcount_y + thread_y; i < true_br; i += tcount_y * tcount_z) {
            const int off_set_i = i * d_k; 
            for (int j = 4 * thread_x; j < d_k; j += 4 * tcount_x) {
                *reinterpret_cast<float4*>(q_i + off_set_i + j) = *reinterpret_cast<float4*>(Q + (i + q_idx * b_r) * d_k + j); 
                *reinterpret_cast<float4*>(o_i + off_set_i + j) = {0, 0, 0, 0}; 
            }
        }
        __syncthreads(); 
        for (int k_idx = 0; k_idx < t_c; k_idx++) {
            int true_bc = min(n_seq_k, (k_idx + 1) * b_c) - b_c * k_idx; 
            for (int i = thread_z * tcount_y + thread_y; i < true_bc; i += tcount_y * tcount_z) {
                for (int j = 4 * thread_x; j < d_k; j += 4 * tcount_x) {
                    // float k_val = K[(i + k_idx * b_c) * d_k + j]; 
                    *reinterpret_cast<float4*>(k_i + i * d_k + j) = *reinterpret_cast<float4*>(K + (i + k_idx * b_c) * d_k + j); 
                    *reinterpret_cast<float4*>(v_i + i * d_k + j) = *reinterpret_cast<float4*>(V + (i + k_idx * b_c) * d_k + j); 
                }
            }
            // __syncthreads();
            // store S in o_i / allocate 32 threads per element
            for (int i = thread_z; i < true_br; i += tcount_z) {
                for (int j = thread_y; j < true_bc; j += tcount_y) {
                    float sum = 0; 
                    // vectorization here
                    for (int k = 4 * thread_x; k < d_k; k += 4 * tcount_x) {
                        float4 q_val = *reinterpret_cast<float4*>(q_i + i * d_k + k); 
                        float4 k_val = *reinterpret_cast<float4*>(k_i + j * d_k + k); 
                        sum += q_val * k_val; 
                    }
                    for (int offset = 4; offset > 0; offset /= 2) {
                        sum += __shfl_down_sync(0xffffffff, sum, offset);
                    }
                    if (thread_x == 0) {
                        o_i[i * d_k + j] = sum; 
                    }
                }
            }
            
        }
    }
}

