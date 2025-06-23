// #include "utils.h"
#include "flash_attention_kernel.h"
#include <assert.h>
#include <cstdio> 
#include <torch/types.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const unsigned full_mask = 0xffffffff; 
const int col_per_thread = 8; 
const int d_k = 64;
// const int sram_size_limit = 49152 / sizeof(float); 
// const int max_b_r = (sram_size_limit / (d_k * 4));  // block rows (query block size)
// const int b_c = b_r; 
// const int block_x = min(32, b_r); 
// const int block_y = 1024 / block_x; 

__forceinline__ __device__ float operator*(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; 
}

__forceinline__ __device__ float4 operator+(const float4 &a, const float4 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; 
}

__forceinline__ __device__ float4 operator*(const float4 &a, const float &b) {
    return {a.x * b, a.y * b, a.z * b, a.w * b}; 
}

__global__ 
void flash_attn_forward(
    float* __restrict__ Q, float* __restrict__ K, float* 
    __restrict__ V, float* __restrict__ O, 
    float* __restrict__ L, float* __restrict__ M, 
    const int b_c, const int b_r, const int t_c, 
    const int t_r, const int n_seq_k, const int n_seq_q,
    const int d_k, const float scaling_factor) {
    int thread_x = threadIdx.x, thread_y = threadIdx.y, thread_z = threadIdx.z; 
    int tcount_x = blockDim.x, tcount_y = blockDim.y, tcount_z = blockDim.z; 
    int ty_lead = (((thread_y * tcount_x + thread_x) % 32) / 8) * 8; 
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
    O += (batch_idx * n_head * n_seq_q * d_k) + head_idx * (n_seq_q * d_k); 
    L += (batch_idx * n_head * n_seq_q) + head_idx * n_seq_q; 
    M += (batch_idx * n_head * n_seq_q) + head_idx * n_seq_q; 

    // https://siboehm.com/articles/22/CUDA-MMM Global Memory Coalescing
    for (int q_idx = 0; q_idx < t_r; q_idx++) {
        int q_idx_br = q_idx * b_r; 
        int true_br = min(n_seq_q, (q_idx + 1) * b_r) - q_idx_br; 
        for (int i = thread_z * tcount_y + thread_y; i < true_br; i += tcount_y * tcount_z) {
            const int offset_i = i * d_k; 
            for (int j = 4 * thread_x; j < d_k; j += 4 * tcount_x) {
                const int offset_2 = offset_i + j; 
                *reinterpret_cast<float4*>(q_i + offset_2) = __ldg(reinterpret_cast<float4*>(Q + (i + q_idx * b_r) * d_k + j)); 
                *reinterpret_cast<float4*>(o_i + offset_2) = {0, 0, 0, 0}; 
            }
        }
        for (int i = 4 * (thread_y * tcount_x + thread_x); i < true_br; i += 4 * tcount_x * tcount_y) {
            *reinterpret_cast<float4*>(L + q_idx_br + i) = {0, 0, 0, 0}; 
            *reinterpret_cast<float4*>(M + q_idx_br + i) = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}; 
        }
        __syncthreads(); 
        float mx = -INFINITY; 
        float sum = 0; 
        for (int k_idx = 0; k_idx < t_c; k_idx++) {
            int true_bc = min(n_seq_k, (k_idx + 1) * b_c) - b_c * k_idx; 
            assert(true_bc % 4 == 0); 
            for (int i = thread_z * tcount_y + thread_y; i < true_bc; i += tcount_y * tcount_z) {
                for (int j = 4 * thread_x; j < d_k; j += 4 * tcount_x) {
                    // float k_val = K[(i + k_idx * b_c) * d_k + j]; 
                    *reinterpret_cast<float4*>(k_i + i * d_k + j) = __ldg(reinterpret_cast<float4*>(K + (i + k_idx * b_c) * d_k + j)); 
                    *reinterpret_cast<float4*>(v_i + i * d_k + j) = __ldg(reinterpret_cast<float4*>(V + (i + k_idx * b_c) * d_k + j)); 
                }
            }
            __syncthreads();
            // // store S in o_i / allocate 32 threads per element
            for (int i = thread_z * tcount_y + thread_y; i < true_br; i += tcount_y * tcount_z) {
                // float mx = M[q_idx_br + i]; 
                // float sum = L[q_idx_br + i]; 
                // printf("%f\n", o_i_im.x); 
                for (int j = 0; j < true_bc; j += col_per_thread) {
                    float dot_prod[col_per_thread]; 
                    for (int c = 0; c < col_per_thread; ++c) dot_prod[c] = 0; 
                    // vectorization + register tiling
                    for (int k = 4 * thread_x; k < d_k; k += 4 * tcount_x) {
                        float4 q_val = *reinterpret_cast<float4*>(q_i + i * d_k + k); 
                        #pragma unroll
                        for (int c = 0; c < col_per_thread; ++c) {
                            float4 k_val = *reinterpret_cast<float4*>(k_i + (j + c) * d_k + k); 
                            dot_prod[c] += q_val * k_val; 
                        }
                    }
                    float new_mx = mx; 
                    #pragma unroll
                    for (int c = 0; c < col_per_thread; ++c) {
                        #pragma unroll
                        for (int offset = 4; offset > 0; offset /= 2) {
                            dot_prod[c] += __shfl_down_sync(full_mask, dot_prod[c], offset);
                        }
                        dot_prod[c] = __shfl_sync(full_mask, dot_prod[c], ty_lead); 
                        float prev_mx = new_mx; 
                        new_mx = fmaxf(new_mx, dot_prod[c]); 
                        sum = sum * __expf(prev_mx - new_mx) + __expf(dot_prod[c] - new_mx); 
                    }
                    
                    for (int c = 0; c < col_per_thread; ++c) {
                        dot_prod[c] = __expf(dot_prod[c] - new_mx); 
                    }

                    float o_i_exp = __expf(mx - new_mx); 

                    for (int l = 0, k = 4 * thread_x; k < d_k; k += 4 * tcount_x, ++l) {
                        float4 v_val = {0, 0, 0, 0}; 
                        for (int c = 0; c < col_per_thread; ++c) {
                            v_val = v_val + *reinterpret_cast<float4*>(v_i + (j + c) * d_k + k) * dot_prod[c]; 
                        }
                        *reinterpret_cast<float4*>(o_i + i * d_k + k) = *reinterpret_cast<float4*>(o_i + i * d_k + k) * o_i_exp + v_val;
                    }
                    mx = new_mx; 
                }
            }
        }
        const float inv_sum = __frcp_rn(sum); 
        for (int i = thread_z * tcount_y + thread_y; i < true_br; i += tcount_y * tcount_z) {
            // float sum = L[q_idx_br + i]; 
            for (int j = 4 * thread_x; j < d_k; j += 4 * tcount_x) {
                *reinterpret_cast<float4*>(O + (q_idx_br + i) * d_k + j) = *reinterpret_cast<float4*>(o_i + i * d_k + j) * inv_sum; 
            }
        }
        if (thread_x == 0) {
            M[q_idx_br + thread_z * tcount_y + thread_y] = mx; 
            L[q_idx_br + thread_z * tcount_y + thread_y] = sum; 
        }
        __syncthreads(); 
    }
}



torch::Tensor forward(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V) {
    cudaFuncSetAttribute(flash_attn_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         65536);

    const int num_batch = Q.size(0); 
    const int num_heads = Q.size(1); 
    const int seq_len_q = Q.size(2); 
    // const int seq_len_q = 1024; 
    const int seq_len_k = K.size(2); 
    const int d_k = K.size(3); 
    const int d_v = d_k; 
    const float scaling_factor = 1.0f/sqrtf(static_cast<float>(d_k)); 
    
    // Flash attention parameters
    // const int b_r = min(seq_len_q, (sram_size_limit / ((d_k + d_v) * 2)));  // block rows (query block size)
    // const int b_c = min(min(d_k, seq_len_k), (sram_size_limit / ((d_k + d_v) * 2)));  // block columns (key/value block size)
    const int offset = -16; 
    const int b_size = 64; 
    const int b_r = b_size - offset; 
    const int b_c = b_size + offset; 
    const int t_r = (seq_len_q + b_r - 1) / b_r;  // number of query tiles
    const int t_c = (seq_len_k + b_c - 1) / b_c;  // number of key tiles
    

    int sram_size = (b_r * d_k + b_c * d_k + b_c * d_v + b_r * d_v) * sizeof(float); // = (b_r + b_c) * (d_k + d_v)

    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto O = torch::empty(Q.sizes(), options);
    float *l, *m ;

    gpuErrchk(cudaMalloc(&l, sizeof(float) * num_batch * num_heads * seq_len_q)); 
    gpuErrchk(cudaMalloc(&m, sizeof(float) * num_batch * num_heads * seq_len_q)); 

    // l = l.to(device); m = m.to(device); 

    dim3 grid(num_batch, num_heads);
    int block_x = 8; 
    int block_y = b_r; 
    int block_z = 1; 
    dim3 block(block_x, block_y, block_z); // Ensure block size doesn't exceed max

    flash_attn_forward<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
        O.data_ptr<float>(), l, m, 
        b_c, b_r, t_c, t_r, seq_len_k, seq_len_q, d_k, scaling_factor
    ); 


    return O; 
}