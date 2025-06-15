#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#pragma once

// CUDA device function declarations

/**
 * Get pointer to item in multi-dimensional array using location indices
 * @param arr: Base array pointer
 * @param loc: Array of [dimension_size, index] pairs for each dimension
 * @param n: Number of dimensions
 * @return: Pointer to the specified element
 */
template <typename T>
__device__ T* _get_item(T* arr, int loc[][2], int n);

/**
 * Load tile data with memory coalescing optimization for warps
 * @param tile: Source tile data
 * @param target: Destination array
 * @param n_rows: Number of rows in tile
 * @param n_cols: Number of columns in tile
 * @param thread_idx: Current thread index
 * @param thread_cnt: Total number of threads
 */
template <typename T> 
__device__ void _load_tile(T* tile, T* target, int n_rows, int n_cols, int thread_idx, int thread_cnt);

/**
 * Perform matrix multiplication with softmax for attention mechanism
 * @param q: Query matrix
 * @param k: Key matrix
 * @param v: Value matrix
 * @param o: Output matrix
 * @param n_seq_q: Sequence length for queries
 * @param n_seq_k: Sequence length for keys
 * @param d_k: Dimension of keys
 * @param thread_cnt: Total number of threads
 * @param q_max: Array to store max values for numerical stability
 * @param q_sum: Array to store sum values for normalization
 */
template <typename T> 
__device__ void _matmul_softmax(T* q, T* k, T* v, T* o, int n_seq_q, int n_seq_k, int d_k, int thread_cnt, T* q_max, T* q_sum);

/**
 * Fill array with specified value using multiple threads
 * @param arr: Array to fill
 * @param size: Size of array
 * @param fill_val: Value to fill with
 * @param thread_cnt: Total number of threads
 * @param thread_idx: Current thread index
 */
template <typename T> 
__device__ void _fill(T* arr, int size, T fill_val, int thread_cnt, int thread_idx);

/**
 * Fill array with specified value using single thread
 * @param arr: Array to fill
 * @param size: Size of array
 * @param value: Value to fill with
 */
template <typename T> 
__device__ void _fill_single_threaded(T* arr, int size, T value);

/**
 * Perform final softmax normalization division
 * @param o_i: Output array to normalize
 * @param n_seq_q: Sequence length
 * @param d_v: Value dimension
 * @param q_sum: Sum values for normalization
 * @param thread_idx: Current thread index
 * @param thread_cnt: Total number of threads
 */
template <typename T> 
__device__ void _softmax_cumdiv(T* o_i, int n_seq_q, int d_v, T* q_sum, int thread_idx, int thread_cnt);

// CUDA math function wrappers for different types
__device__ inline float maxf(float a, float b) {
    return fmaxf(a, b);
}

__device__ inline double maxf(double a, double b) {
    return fmax(a, b);
}

__device__ inline float exp(float x) {
    return expf(x);
}

__device__ inline double exp(double x) {
    return ::exp(x);
}

#endif // UTILS_H