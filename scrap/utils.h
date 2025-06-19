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
__device__ float* _get_item(float* arr, int loc[][2], int n);

/**
 * Load tile data with memory coalescing optimization for warps
 * @param source: Source tile data
 * @param target: Destination array
 * @param n: amount of columns in source
 * @param start_row: Number of rows in tile
 * @param end_row: Number of columns in tile
 * @param start_col: Current thread index
 * @param end_col: Total number of threads
 * @param thread_idx: Thread index
 * @param thread_cnt: Amount of threads in the block
 */
__device__ void _load_tile(float* source, float* target, int n, int start_row, int end_row, int start_col, int end_col, int thread_idx, int thread_cnt); 

/**
 * Perform matrix multiplication with softmax for attention mechanism
 * @param q: Query matrix
 * @param k: Key matrix
 * @param v: Value matrix
 * @param o: Output matrix
 * @param n_seq_q: Sequence length for queries
 * @param n_seq_k: Sequence length for keys
 * @param d_k: Dimension of keys
 * @param d_v: Dimension of values
 * @param thread_cnt: Total number of threads
 * @param thread_idx: Current thread index
 * @param q_max: Array to store max values for numerical stability
 * @param q_sum: Array to store sum values for normalization
 */
__device__ void _matmul_softmax(float* q, float* k, float* v, float* o, int n_seq_q, int n_seq_k, int d_k, int d_v, int thread_cnt, int thread_idx, float* q_max, float* q_sum, float scaling_factor);

/**
 * Fill array with specified value using multiple threads
 * @param arr: Array to fill
 * @param size: Size of array
 * @param fill_val: Value to fill with
 * @param thread_cnt: Total number of threads
 * @param thread_idx: Current thread index
 */
__device__ void _fill(float* arr, int size, float fill_val, int thread_cnt, int thread_idx);

/**
 * Fill array with specified value using single thread
 * @param arr: Array to fill
 * @param size: Size of array
 * @param value: Value to fill with
 */
__device__ void _fill_single_threaded(float* arr, int size, float value);

/**
 * Perform final softmax normalization division
 * @param o_i: Output array to normalize
 * @param n_seq_q: Sequence length
 * @param d_v: Value dimension
 * @param q_sum: Sum values for normalization
 * @param thread_idx: Current thread index
 * @param thread_cnt: Total number of threads
 */
__device__ void _softmax_cumdiv(float* o_i, int n_seq_q, int d_v, float* q_sum, int thread_idx, int thread_cnt);

#endif // UTILS_H
