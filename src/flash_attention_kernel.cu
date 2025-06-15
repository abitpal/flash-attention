#include "utils.h"
#include "flash_attention_kernel.h"
#include "consts.h"


__global__ 
void flash_attn_forward(float* Q, float* K, float* V, flash_attn_forward_params* faf_param) {
    /*
        Input: 
            Q : Pointer to Q^T with shape (batch, num_head, d_k, num_seq_q) - lives in contiguous memory
            K : Pointer to K^T with shape (batch, num_head, d_k, num_seq_k) - lives in contiguous memory
            V : Pointer to V^T with shape (batch, num_head, d_v, num_seq_k) - lives in contiguous memory
        Return: 
            None - Modifies O
    */
    int tidx = threadIdx.x; // index of the specific row, d_idx < d_k
    int tcount = blockDim.x; 
    int batch_idx = blockIdx.x, head_idx = blockIdx.y;  // batch and head index
    int n_head = gridDim.y; 

    const int b_c = faf_param->b_c, b_r = faf_param->b_r, t_c = faf_param->t_c, t_r = faf_param->t_r; 
    const int d_k = faf_param->d_k, d_v = faf_param->d_v; 
    const int n_seq_k = faf_param->n_seq_k, n_seq_q = faf_param->n_seq_q; 

    int queries_per_thread = (d_k + tcount - 1) / tcount; // not on sram --> storage on register

    extern __shared__ float sram []; 
    float* q_i = sram; 
    float* k_i = sram + b_r * d_k; 
    float* v_i = k_i + b_c * d_k; 
    float* o_i = v_i + b_c * d_v; 

    // https://siboehm.com/articles/22/CUDA-MMM Global Memory Coalescing

    for (int q_idx = 0; q_idx < t_r; q_idx++) {
        // load q_i
        int loc_q_tile[4][2] = {{-1, batch_idx},{n_head, head_idx},{d_k, 0},{n_seq_q, q_idx * b_r}};
        float* q_tile = _get_item(Q, loc_q_tile, 4); 
        _load_tile(q_tile, q_i, d_k, b_r, tidx, tcount); // this is a warp-aware load that loads w/ memory coalescing
        // init o_i to 0
        _fill(o_i, b_r * d_v, 0.0f, tcount, tidx); 
        float q_max[max_queries_per_thread], q_sum[max_queries_per_thread]; 
        _fill_single_threaded(q_max, queries_per_thread, -INFINITY); 
        _fill_single_threaded(q_sum, queries_per_thread, 0.0f); 
        for (int k_idx = 0; k_idx < t_c; k_idx++) {
            // load k into sram
            int loc_k_tile[4][2] = {{-1, batch_idx}, {n_head, head_idx}, {d_k, 0}, {n_seq_k, k_idx * b_c}}; 
            int loc_v_tile[4][2] =  {{-1, batch_idx}, {n_head, head_idx}, {d_v, 0}, {n_seq_k, k_idx * b_c}}; 
            float* k_tile = _get_item(K, loc_k_tile, 4); 
            float* v_tile = _get_item(V, loc_v_tile, 4); 
            _load_tile(k_tile, k_i, d_k, b_c, tidx, tcount); 
            _load_tile(v_tile, v_i, d_k, b_c, tidx, tcount); 
            __syncthreads(); 
            // compute Q * K
            _matmul_softmax(q_i, k_i, v_i, o_i, b_r, b_c, d_k, d_v, tcount, tidx, q_max, q_sum); 
        }
        // final update on o_i
        _softmax_cumdiv(o_i, b_r, d_v, q_sum, tidx, tcount); 
        __syncthreads(); 
    }
}

