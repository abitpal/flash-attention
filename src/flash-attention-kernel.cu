#include "utils.h"

__global__ 
void scaled_dot_product_attention_kernel(const _Float16* Q, const _Float16* K, const _Float16* V, const int d_k, const int d_v) {
    /*
        Input: 
            Q : Pointer to Q^T with shape (batch, num_head, d_k, num_seq_q) - lives in contiguous memory
            K : Pointer to K^T with shape (batch, num_head, d_k, num_seq_k) - lives in contiguous memory
            V : Pointer to V^T with shape (batch, num_head, num_seq_k, d_v) - lives in contiguous memory
        Return: 
            None - Modifies O
    */
    int tidx = threadIdx.x; // index of the specific row, d_idx < d_k
    int tcount = blockDim.x; 
    int batch_idx = blockIdx.x, head_idx = blockIdx.y;  // batch and head index
    int n_head = gridDim.y; 

    extern int b_c, b_r, t_c, t_r; 
    extern int n_seq_k, n_seq_q; 

    extern __shared__ _Float16 sram []; 
    _Float16* q_i = sram; 
    _Float16* k_i = sram + br * d_k; 

    // https://siboehm.com/articles/22/CUDA-MMM Global Memory Coalescing

    for (int q_idx = 0; q_idx < t_r; q_idx++) {
        // load q_i
        _Float16* q_tile = _get_item(Q, {{-1, batch_idx}, {n_head, head_idx}, {d_k, 0}, {n_seq_q, q_idx * b_r}}); 
        _load_tile(q_tile, q_i, d_k, br, tidx, tcount); // this is a warp-aware load that loads w/ memory coalescing
        for (int k_idx = 0; k_idx < t_c; k_idx++) {
            // load k into sram
            _Float16* k_tile = _get_item(K, {{-1, batch_idx}, {n_head, head_idx}, {d_k, 0}, {n_seq_k, k_idx * b_c}}); 
            _load_tile(k_tile, k_i, d_k, bc, tidx, tcount); 
            __syncthreads(); 
            // compute Q * K
        }
    }
}