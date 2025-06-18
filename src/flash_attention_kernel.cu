#include "utils.h"
#include "flash_attention_kernel.h"
#include "consts.h"
#include <assert.h>
#include <cstdio> 

__global__ 
void flash_attn_forward(
    float* Q, float* K, float* V, float* O, const int b_c, const int b_r, const int t_c, 
    const int t_r, const int n_seq_k, const int n_seq_q,
    const int d_k, const float scaling_factor) {
    int thread_x = threadIdx.x, thread_y = threadIdx.y; 
    int tcount_x = blockDim.x, tcount_y = blockDim.y; 
    int tid = thread_y * tcount_x + thread_x; 
    int batch_idx = blockIdx.x, head_idx = blockIdx.y;  // batch and head index
    int n_head = gridDim.y; 

    extern __shared__ float sram []; 
    float* q_i = sram; 
    float* k_i = sram + b_r * d_k; 
    float* v_i = k_i + b_c * d_k; 
    float* o_i = v_i + b_c * d_k; 

    Q += (batch_idx * n_head * n_seq_q * d_k) + head_idx * (n_seq_q * d_k); 

    // https://siboehm.com/articles/22/CUDA-MMM Global Memory Coalescing
    for (int q_idx = 0; q_idx < t_r; q_idx++) {
        int true_br = min(n_seq_q, (q_idx + 1) * b_r) - b_r * q_idx; 
        for (int i = thread_y; i < true_br; i += tcount_y) {
            int off_set_i = i * d_k; 
            for (int j = thread_x; j < d_k; j += tcount_x) {
                q_i[off_set_i + j] = Q[(i + q_idx * b_r) * d_k + j];
                o_i[off_set_i + j] = 0; 
            }
        }
        __syncthreads(); 

        for (int k_idx = 0; k_idx < t_c; k_idx++) {
            int true_bc = min(n_seq_k, (k_idx + 1) * b_c) - b_c * k_idx; 
            for (int i = thread_y; i < true_bc; i += tcount_y) {
                for (int j = thread_x; j < d_k; j += tcount_x) {
                    // float k_val = K[(i + k_idx * b_c) * d_k + j]; 
                    k_i[i * d_k + j] = K[(i + k_idx * b_c) * d_k + j]; 
                    v_i[i * d_k + j] = V[(i + k_idx * b_c) * d_k + j]; 
                }
            }
            __syncthreads();
            for (int i = thread_y; i < true_br; i += tcount_y) {
                for (int j = 0; j < true_bc; j++) {
                    float sum = 0; 
                    for (int k = 0; k < d_k; k++) {
                        sum += k_i[j * d_k + k] * q_i[i * d_k + k]; 
                    }
                    for (int k = thread_x; k < d_k; k+=tcount_x) {
                        o_i[i * d_k + k] += sum * v_i[j * d_k + k]; 
                    }
                }
            }
            // for (int i = 0; i < true_bc; i++) {
            //     for (int j = thread_y; j < true_bc; j += tcount_y) {
            //         for (int k = thread_x; k < true_)
            //     }
            // }

        }
    }
}

