#include <cuda_profiler_api.h> // at the top
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include "flash_attention_kernel.h"
#include <cmath>

// Helper function to check CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Helper function to initialize random data
template <typename T>
void init_random_data(T* data, int size, T scale = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, scale);
    
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<T>(dis(gen));
    }
}

template <typename T>
void reference_attention(const T* Q, const T* K, const T* V, T* O_ref,
                        int batch_size, int num_heads, int seq_len_q, int seq_len_k, 
                        int d_k, int d_v) {
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Compute attention scores: Q * K^T
            std::vector<T> scores(seq_len_q * seq_len_k, 0);
            
            for (int i = 0; i < seq_len_q; i++) {
                for (int j = 0; j < seq_len_k; j++) {
                    T dot_product = 0;
                    for (int k = 0; k < d_k; k++) {
                        // Updated indexing for shape (b, nh, seq_len, d_k)
                        int q_idx = b * num_heads * seq_len_q * d_k + h * seq_len_q * d_k + i * d_k + k;
                        int k_idx = b * num_heads * seq_len_k * d_k + h * seq_len_k * d_k + j * d_k + k;
                        dot_product += Q[q_idx] * K[k_idx];
                    }
                    scores[i * seq_len_k + j] = dot_product / sqrt(static_cast<T>(d_k));
                }
            }
            
            // Apply softmax
            for (int i = 0; i < seq_len_q; i++) {
                T max_val = scores[i * seq_len_k];
                for (int j = 1; j < seq_len_k; j++) {
                    max_val = std::max(max_val, scores[i * seq_len_k + j]);
                }
                
                T sum_exp = 0;
                for (int j = 0; j < seq_len_k; j++) {
                    scores[i * seq_len_k + j] = exp(scores[i * seq_len_k + j] - max_val);
                    sum_exp += scores[i * seq_len_k + j];
                }
                
                for (int j = 0; j < seq_len_k; j++) {
                    scores[i * seq_len_k + j] /= sum_exp;
                }
            }
            
            // Compute output: scores * V
            for (int i = 0; i < seq_len_q; i++) {
                for (int k = 0; k < d_v; k++) {
                    T output_val = 0;
                    for (int j = 0; j < seq_len_k; j++) {
                        // Updated indexing for shape (b, nh, seq_len, d_v)
                        int v_idx = b * num_heads * seq_len_k * d_v + h * seq_len_k * d_v + j * d_v + k;
                        output_val += scores[i * seq_len_k + j] * V[v_idx];
                    }
                    // Updated indexing for output shape (b, nh, seq_len, d_v)
                    int o_idx = b * num_heads * seq_len_q * d_v + h * seq_len_q * d_v + i * d_v + k;
                    O_ref[o_idx] = output_val;
                }
            }
        }
    }
}

// Function to compare results
template <typename T>
bool compare_results(const T* gpu_result, const T* cpu_result, int size, T tolerance = 1e-3) {
    T max_diff = 0;
    int max_diff_idx = 0;
    
    for (int i = 0; i < size; i++) {
        T diff = std::abs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    std::cout << "Max difference: " << max_diff << " at index " << max_diff_idx << std::endl;
    std::cout << "GPU value: " << gpu_result[max_diff_idx] << ", CPU value: " << cpu_result[max_diff_idx] << std::endl;
    
    return max_diff < tolerance;
}

int main() {

    int device = 0; 
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    int sram_size_limit = prop.sharedMemPerBlock / sizeof(float); 

    // Test parameters - 4, 8, 128, 64
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_len_q = 128; 
    // const int seq_len_q = 1024; 
    const int seq_len_k = seq_len_q; 
    const int d_k = 64;
    const int d_v = 64;
    const float scaling_factor = 1.0f/sqrtf(static_cast<float>(d_k)); 
    
    // Flash attention parameters
    // const int b_r = min(seq_len_q, (sram_size_limit / ((d_k + d_v) * 2)));  // block rows (query block size)
    // const int b_c = min(min(d_k, seq_len_k), (sram_size_limit / ((d_k + d_v) * 2)));  // block columns (key/value block size)
    const int offset = -16; 
    const int b_r = 48 - offset; 
    const int b_c = 48 + offset; 
    const int t_r = (seq_len_q + b_r - 1) / b_r;  // number of query tiles
    const int t_c = (seq_len_k + b_c - 1) / b_c;  // number of key tiles

    int sram_size = (b_r * d_k + b_c * d_k + b_c * d_v + b_r * d_v) * sizeof(float); // = (b_r + b_c) * (d_k + d_v)
    /*
    sram_size / (d_k + d_v) / 2
    */
    std::cout << "Shared memory size: " << sram_size << " bytes" << std::endl;
    assert((sram_size <= prop.sharedMemPerBlock)); 
    
    std::cout << "Test Parameters:" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Num heads: " << num_heads << std::endl;
    std::cout << "Seq len Q: " << seq_len_q << ", Seq len K: " << seq_len_k << std::endl;
    std::cout << "d_k: " << d_k << ", d_v: " << d_v << std::endl;
    std::cout << "Block size (b_r, b_c): (" << b_r << ", " << b_c << ")" << std::endl;
    std::cout << "Tiles (t_r, t_c): (" << t_r << ", " << t_c << ")" << std::endl;
    
    // Calculate sizes
    const int q_size = batch_size * num_heads * d_k * seq_len_q;
    const int k_size = batch_size * num_heads * d_k * seq_len_k;
    const int v_size = batch_size * num_heads * d_v * seq_len_k;
    const int o_size = batch_size * num_heads * d_v * seq_len_q;
    
    // Allocate host memory
    std::vector<float> h_Q(q_size);
    std::vector<float> h_K(k_size);
    std::vector<float> h_V(v_size);
    std::vector<float> h_O(o_size);
    std::vector<float> h_O_ref(o_size);
    
    // Initialize input data
    std::cout << "Initializing input data..." << std::endl;
    init_random_data(h_Q.data(), q_size, 0.1f);
    init_random_data(h_K.data(), k_size, 0.1f);
    init_random_data(h_V.data(), v_size, 0.1f);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O; 
    flash_attn_forward_params *d_params;
    
    CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, v_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, o_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(flash_attn_forward_params)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), k_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), v_size * sizeof(float), cudaMemcpyHostToDevice));
            
    // Launch kernel
    std::cout << "Launching Flash Attention kernel..." << std::endl;
    
    dim3 grid(batch_size, num_heads);
    int block_x = 8; 
    int block_y = b_r; 
    int block_z = 1; 
    dim3 block(block_x, block_y, block_z); // Ensure block size doesn't exceed max

    // std::cout << "Block dim: " << 1024 / b_r << ' ' << b_r << '\n'; 

    // std::cout << "Key:\n"; 
    // for (int i = 0; i < d_k; i++) {
    //     for (int j = 0; j < seq_len_q; j++) {
    //         std::cout << h_K[i * seq_len_q + j] << ' '; 
    //     }
    //     std::cout << '\n'; 
    // }
    // std::cout << '\n'; 
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    cudaProfilerStart();  // Begin nsys profiling window

    CUDA_CHECK(cudaEventRecord(start));
    flash_attn_forward<<<grid, block, sram_size>>>(d_Q, d_K, d_V, d_O, b_c, b_r, t_c, t_r, seq_len_k, seq_len_q, d_k, scaling_factor);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel to finish
    cudaProfilerStop();  // End nsys profiling window

    CUDA_CHECK(cudaGetLastError());  // Check for errors
        
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, o_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute reference result
    // std::cout << "Computing reference result..." << std::endl;
    // reference_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
    //                    batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v);
    
    // // Compare results
    // std::cout << "Comparing results..." << std::endl;
    // bool passed = compare_results(h_O.data(), h_O_ref.data(), o_size, 1e-2f);
    
    // if (passed) {
    //     std::cout << "Test PASSED!" << std::endl;
    // } else {
    //     std::cout << "Test FAILED!" << std::endl;
    // }
    
    // // Print some sample values for inspection
    // std::cout << "\nSample values comparison:" << std::endl;
    // for (int i = 0; i < std::min(10, o_size); i++) {
    //     std::cout << "Index " << i << ": GPU=" << h_O[i] << ", CPU=" << h_O_ref[i] << std::endl;
    // }

    // print O

    // for (int i = 0; i < d_k; i++) {
    //     for (int j = 0; j < seq_len_q; j++) {
    //         printf("%.5f=%.5f ", h_O[i * seq_len_q + j], h_O_ref[i * seq_len_q + j]); 
    //     }
    //     std::cout << '\n'; 
    // }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0; 
}