#include "utils.h"

template <typename T>
__device__ T* _get_item(T* arr, int[][2] loc, int n) {
    int it = 0; 
    int cur = 1; 
    for (int i = n - 1; i >= 0; i--) {
        it += cur * loc[i][1];  
        cur *= loc[i][0]; 
    }
    return arr + it; 
}

/*
Making sure we have memory coalescing within warps
*/
template <typename T> 
__device__ _load_tile(T* tile, T* target, int n_rows, int n_cols, int thread_idx, int thread_cnt) {
    int ld_amt_per_thread = (n_cols + thread_cnt - 1) / thread_cnt; 
    for (int i = 0; i < n_rows; i++) {
        for (int j = thread_idx; j < n_cols; j += thread_cnt) {
            int idx = i * n_cols + j; 
            target[idx] = tile[idx]; 
        }
    }
}

template <typename T> 
__device__ _matmul_softmax(T* q, T* k, T* v, int n_seq_q, int n_seq_k, int d_k, int thread_cnt, T* q_max, T* q_sum) {
    int queries_per_thread = (d_k + thread_cnt - 1) / thread_cnt; 

    for (int i = 0; i < n_seq_k; i++) {
        T dot_prod[queries_per_thread]; 
        for (int j = 0; j < queries_per_thread; j++) dot_prod[j] = 0.0f; 
        for (int j = 0; j < d_k; j++) {
            T k_val = k[n_seq_k * j + i]; 
            for (int k = thread_idx, int section = 0; k < n_seq_q; k += thread_cnt, ++section) {
                T q_val = q[n_seq_q * j + k]; 
                dot_prod[section] += k_val * q_val; 
            }
        }
        
        for (int j = 0; j < queries_per_thread; j++) {
            T prev_max = q_max[j]; 
            q_max[j] = maxf(q_max[j], dot_prod[j]); 
            q_sum[j] = q_sum[j] * exp(prev_max - q_max[j]) + exp(dot_prod[j] - q_max[j]); 
        }
    }
}
