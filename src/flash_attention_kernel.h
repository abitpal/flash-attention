/**
 * @file flash_attn_forward.h
 * @brief Header for FlashAttention forward kernel and parameter struct.
 *
 * This file defines the parameter struct and CUDA kernel interface for 
 * the forward pass of FlashAttention, a memory-efficient attention mechanism 
 * optimized for high throughput and scalability.
 *
 * The input tensors Q, K, and V are assumed to be in column-major (transposed) format:
 *     - Q^T ∈ (batch_size, num_heads, d_k, n_seq_q)
 *     - K^T ∈ (batch_size, num_heads, d_k, n_seq_k)
 *     - V^T ∈ (batch_size, num_heads, d_v, n_seq_k)
 *
 * Each kernel block is responsible for computing a tile of the output matrix O
 * corresponding to a chunk of queries and all keys, using shared memory tiling
 * and warp-level memory coalescing techniques.
 */

#pragma once

/**
 * @struct flash_attn_forward_params
 * @brief Tiling and dimensional parameters for FlashAttention forward kernel.
 *
 * This structure encapsulates the key dimensions and tile/blocking parameters 
 * required by the FlashAttention forward kernel.
 */
typedef struct {
    const int d_k;       ///< Dimensionality of keys and queries (dot product size)
    const int d_v;       ///< Dimensionality of values (output feature size)
    const int b_c;       ///< Tile width (number of key/value columns per tile)
    const int b_r;       ///< Tile height (number of query rows per tile)
    const int t_c;       ///< Number of tiles along the key/value sequence (K/V columns)
    const int t_r;       ///< Number of tiles along the query sequence (Q rows)
    const int n_seq_k;   ///< Total key/value sequence length
    const int n_seq_q;   ///< Total query sequence length
} flash_attn_forward_params; 

/**
 * @brief FlashAttention forward CUDA kernel (templated).
 *
 * This kernel performs the forward pass of scaled dot-product attention using
 * a shared-memory tiling strategy optimized for high throughput.
 *
 * @param Q Pointer to query matrix Q^T of shape (B, H, d_k, n_seq_q)
 * @param K Pointer to key matrix K^T of shape (B, H, d_k, n_seq_k)
 * @param V Pointer to value matrix V^T of shape (B, H, d_v, n_seq_k)
 * @param faf_param Pointer to a struct containing tiling and dimension parameters
 *
 * @note Output matrix O is assumed to be written to global memory
 *       within the kernel (not shown in this declaration).
 */
__global__ 
void flash_attn_forward(float* Q, float* K, float* V, flash_attn_forward_params* faf_param);
