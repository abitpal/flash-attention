#include "utils.h"
#include "consts.h"
#include <cmath>

__device__ float* _get_item(float* arr, int loc[][2], int n) {
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
__device__ void _load_tile(float* source, float* target, int n, int start_row, int end_row, int start_col, int end_col, int thread_idx, int thread_cnt) {
    int n_cols = end_col - start_col + 1; 
    int n_rows = end_row - start_row + 1; 
    int ld_amt_per_thread = (n_cols + thread_cnt - 1) / thread_cnt; 
    for (int i = 0; i < n_rows; i++) {
        for (int j = thread_idx; j < n_cols; j += thread_cnt) {
            int idx = i * n_cols + j; 
            target[idx] = source[(start_row + i) * n + start_col + j]; 
        }
    }
}

__device__ void _matmul_softmax(
    float* q, float* k, float* v, float* o,
    int n_seq_q, int n_seq_k, int d_k, int d_v,
    int thread_cnt, int thread_idx,
    float* q_max, float* q_sum, float scaling_factor=1.0f
) {
    int queries_per_thread = (d_k + thread_cnt - 1) / thread_cnt; 
    for (int i = 0; i < n_seq_k; i++) {
        float dot_prod[max_queries_per_thread];  
        for (int j = 0; j < queries_per_thread; j++) dot_prod[j] = 0.0f; 
        for (int j = 0; j < d_k; j++) {
            float k_val = k[n_seq_k * j + i]; 
            for (int k = thread_idx, section = 0; k < n_seq_q; k += thread_cnt, ++section) {
                float q_val = q[n_seq_q * j + k]; 
                dot_prod[section] += k_val * q_val; 
            }
        }
        for (int j = 0, q_col = thread_idx; j < queries_per_thread && q_col < n_seq_q; j++, q_col += thread_cnt) {
            dot_prod[j] = dot_prod[j] * scaling_factor; 
            float prev_max = q_max[j]; 
            q_max[j] = fmaxf(q_max[j], dot_prod[j]); 
            q_sum[j] = q_sum[j] * expf(prev_max - q_max[j]) + expf(dot_prod[j] - q_max[j]); 
            for (int k = 0; k < d_v; k++) {
                o[n_seq_q * k + q_col] = 
                    o[n_seq_q * k + q_col] * expf(prev_max - q_max[j]) +
                    expf(dot_prod[j] - q_max[j]) * v[k * n_seq_k + i]; 
            }
        }
    }
}

__device__ void _fill(float* arr, int size, float fill_val, int thread_cnt, int thread_idx) {
    int val_per_thread = (size + thread_cnt - 1) / thread_cnt; 
    for (int i = thread_idx; i < size; i += thread_cnt) {
        arr[i] = fill_val; 
    }
}

__device__ void _fill_single_threaded(float* arr, int size, float value) {
    for (int i = 0; i < size; i++) {
        arr[i] = value; 
    }
}

__device__ void _softmax_cumdiv(float* o_i, int n_seq_q, int d_v, float* q_sum, int thread_idx, int thread_cnt) {
    for (int i = 0; i < d_v; i++) {
        for (int j = thread_idx, k = 0; j < n_seq_q; j += thread_cnt, ++k) {
            o_i[i * n_seq_q + j] /= q_sum[k]; 
        }
    }
}
